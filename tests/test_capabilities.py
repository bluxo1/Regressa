from __future__ import annotations

import pytest

from regressa import capabilities


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(capabilities.UnknownModelError):
        capabilities.get("gpt-4-turbo")


def test_mock_models_resolve_to_a_zero_cost_entry():
    caps = capabilities.get("mock-anything")
    assert caps.input_per_mtok == 0.0
    assert caps.supports_sampling is True


def test_price_uses_per_mtok_rates():
    # 1M in + 1M out on Opus 5 = $5 + $25.
    assert capabilities.price("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_cache_reads_are_a_tenth_of_input():
    cost = capabilities.price("claude-opus-5", 0, 0, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(0.5)


def test_temperature_is_rejected_on_models_that_400_on_it():
    issues = capabilities.validate_target("t", "claude-opus-5", temperature=0.0)
    assert len(issues) == 1
    assert "temperature" in issues[0].message


def test_temperature_is_allowed_on_models_that_accept_it():
    assert capabilities.validate_target("t", "claude-sonnet-4-6", temperature=0.2) == []


def test_disabled_thinking_above_high_effort_is_rejected_on_opus_5():
    issues = capabilities.validate_target(
        "t", "claude-opus-5", thinking="disabled", effort="xhigh"
    )
    assert len(issues) == 1
    assert "effort" in issues[0].message


def test_disabled_thinking_at_high_effort_is_fine_on_opus_5():
    assert (
        capabilities.validate_target("t", "claude-opus-5", thinking="disabled", effort="high")
        == []
    )


def test_fable_rejects_any_explicit_thinking_config():
    issues = capabilities.validate_target("t", "claude-fable-5", thinking="disabled")
    assert len(issues) == 1
    assert "always on" in issues[0].message


def test_effort_is_rejected_on_models_without_the_parameter():
    issues = capabilities.validate_target("t", "claude-haiku-4-5", effort="high")
    assert len(issues) == 1
    assert "does not support `effort`" in issues[0].message


def test_max_tokens_ceiling_is_enforced():
    issues = capabilities.validate_target("t", "claude-haiku-4-5", max_tokens=100_000)
    assert len(issues) == 1
    assert "ceiling" in issues[0].message
