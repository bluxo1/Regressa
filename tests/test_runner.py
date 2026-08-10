from __future__ import annotations

import pytest

from regressa.config import Suite
from regressa.providers.anthropic_provider import AnthropicProvider
from regressa.providers.base import CompletionRequest
from regressa.runner import ConfigError, Runner, RunOptions


def test_runs_every_case_against_every_target(mock_suite, store):
    run = Runner(store).run(mock_suite, RunOptions(concurrency=1))
    assert run.total == 2
    assert run.passed == 2
    assert run.score == pytest.approx(1.0)


def test_results_are_sorted_regardless_of_completion_order(mock_suite, store):
    run = Runner(store).run(mock_suite, RunOptions(concurrency=4))
    keys = [r.key for r in run.results]
    assert keys == sorted(keys)


def test_failing_assertion_marks_the_case_failed(store):
    suite = Suite.model_validate(
        {
            "name": "s",
            "targets": [
                {
                    "id": "m",
                    "provider": "mock",
                    "model": "mock-1",
                    "options": {"default": "nope"},
                }
            ],
            "cases": [
                {
                    "id": "c",
                    "prompt": "p",
                    "assertions": [{"type": "contains", "value": "expected"}],
                }
            ],
        }
    )
    run = Runner(store).run(suite)
    assert run.failed == 1
    assert run.errors == 0


def test_provider_failure_is_recorded_as_an_error_not_a_regression(store):
    suite = Suite.model_validate(
        {
            "name": "s",
            "targets": [
                {
                    "id": "m",
                    "provider": "mock",
                    "model": "mock-1",
                    "options": {"fail": "simulated network failure"},
                }
            ],
            "cases": [
                {"id": "c", "prompt": "p", "assertions": [{"type": "json_valid"}]}
            ],
        }
    )
    run = Runner(store).run(suite)
    assert run.errors == 1
    assert run.results[0].error == "simulated network failure"
    assert run.results[0].passed is False


def test_a_target_that_would_400_is_rejected_before_any_call(store):
    suite = Suite.model_validate(
        {
            "name": "s",
            "targets": [
                {"id": "t", "model": "claude-opus-5", "temperature": 0.0},
            ],
            "cases": [{"id": "c", "prompt": "p"}],
        }
    )
    with pytest.raises(ConfigError) as exc:
        Runner(store).run(suite)
    assert "temperature" in str(exc.value)


def test_missing_prompt_variable_is_a_config_error(store):
    suite = Suite.model_validate(
        {
            "name": "s",
            "targets": [
                {"id": "m", "provider": "mock", "model": "mock-1",
                 "options": {"default": "x"}}
            ],
            "cases": [{"id": "c", "prompt": "Hello {{ missing }}"}],
        }
    )
    with pytest.raises(ConfigError) as exc:
        Runner(store).run(suite)
    assert "missing" in str(exc.value)


def test_weighted_score_favours_heavier_assertions(store):
    suite = Suite.model_validate(
        {
            "name": "s",
            "targets": [
                {"id": "m", "provider": "mock", "model": "mock-1",
                 "options": {"default": "alpha"}}
            ],
            "cases": [
                {
                    "id": "c",
                    "prompt": "p",
                    "assertions": [
                        {"type": "contains", "value": "alpha", "weight": 3},
                        {"type": "contains", "value": "beta", "weight": 1},
                    ],
                }
            ],
        }
    )
    run = Runner(store).run(suite)
    # 3 of 4 weight units passed.
    assert run.results[0].score == pytest.approx(0.75)
    assert run.results[0].passed is False


def test_tag_and_case_filters(mock_suite, store):
    runner = Runner(store)
    run = runner.run(mock_suite, RunOptions(cases=["hello"]))
    assert [r.case_id for r in run.results] == ["hello"]

    with pytest.raises(ConfigError):
        runner.run(mock_suite, RunOptions(tags=["nonexistent"]))


def test_case_can_restrict_itself_to_a_subset_of_targets(store):
    suite = Suite.model_validate(
        {
            "name": "s",
            "targets": [
                {"id": "a", "provider": "mock", "model": "mock-1",
                 "options": {"default": "x"}},
                {"id": "b", "provider": "mock", "model": "mock-1",
                 "options": {"default": "x"}},
            ],
            "cases": [{"id": "c", "prompt": "p", "targets": ["b"]}],
        }
    )
    run = Runner(store).run(suite)
    assert [r.target_id for r in run.results] == ["b"]


def test_suite_defaults_are_folded_into_targets():
    suite = Suite.model_validate(
        {
            "name": "s",
            "defaults": {"system": "shared system", "effort": "low"},
            "targets": [{"id": "a", "model": "claude-opus-5"}],
            "cases": [{"id": "c", "prompt": "p"}],
        }
    )
    resolved = suite.resolved_targets(suite.cases[0])[0]
    assert resolved.system == "shared system"
    assert resolved.effort == "low"


def test_default_assertions_apply_to_every_case():
    suite = Suite.model_validate(
        {
            "name": "s",
            "defaults": {"assertions": [{"type": "not_contains", "value": "leak"}]},
            "targets": [{"id": "a", "model": "claude-opus-5"}],
            "cases": [
                {"id": "c", "prompt": "p",
                 "assertions": [{"type": "contains", "value": "x"}]}
            ],
        }
    )
    assert len(suite.resolved_assertions(suite.cases[0])) == 2


# --------------------------------------------------------------------------- #
# Provider request construction — the capability gating most likely to regress.
# --------------------------------------------------------------------------- #


def _request(model: str, **kwargs) -> CompletionRequest:
    return CompletionRequest(model=model, prompt="hi", max_tokens=100, **kwargs)


def test_temperature_is_never_sent_to_a_model_that_rejects_it():
    params = AnthropicProvider().build_params(_request("claude-opus-5", temperature=0.5))
    assert "temperature" not in params


def test_temperature_is_sent_to_a_model_that_accepts_it():
    params = AnthropicProvider().build_params(_request("claude-sonnet-4-6", temperature=0.5))
    assert params["temperature"] == 0.5


def test_effort_is_nested_under_output_config():
    params = AnthropicProvider().build_params(_request("claude-opus-5", effort="high"))
    assert params["output_config"] == {"effort": "high"}


def test_effort_is_omitted_for_models_without_the_parameter():
    params = AnthropicProvider().build_params(_request("claude-haiku-4-5", effort="high"))
    assert "output_config" not in params


def test_thinking_is_omitted_unless_configured():
    params = AnthropicProvider().build_params(_request("claude-opus-5"))
    assert "thinking" not in params


def test_thinking_disabled_is_never_sent_to_fable():
    params = AnthropicProvider().build_params(_request("claude-fable-5", thinking="disabled"))
    assert "thinking" not in params


def test_cache_key_ignores_provider_options_and_case_id():
    a = _request("claude-opus-5", options={"x": 1}, case_id="one")
    b = _request("claude-opus-5", options={"y": 2}, case_id="two")
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_with_the_system_prompt():
    a = _request("claude-opus-5", system="one")
    b = _request("claude-opus-5", system="two")
    assert a.cache_key() != b.cache_key()
