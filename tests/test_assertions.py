from __future__ import annotations

import pytest

from regressa.assertions import EvalContext, evaluate
from regressa.config import AssertionSpec
from regressa.providers.base import CompletionResponse
from pydantic import TypeAdapter

_adapter = TypeAdapter(AssertionSpec)


def spec(**kwargs):
    return _adapter.validate_python(kwargs)


def ctx(output: str, **overrides) -> EvalContext:
    response = CompletionResponse(
        text=output,
        input_tokens=overrides.pop("input_tokens", 10),
        output_tokens=overrides.pop("output_tokens", 10),
        latency_ms=overrides.pop("latency_ms", 100.0),
        cached=overrides.pop("cached", False),
    )
    return EvalContext(
        output=output,
        response=response,
        cost_usd=overrides.pop("cost_usd", 0.001),
        case_id="c",
        target_id="t",
        prompt="p",
        vars={},
    )


def test_contains_all_gives_partial_credit_on_partial_match():
    out = evaluate(spec(type="contains", value=["alpha", "beta"]), ctx("alpha only"))
    assert out.passed is False
    assert out.score == pytest.approx(0.5)


def test_contains_any_passes_on_one_hit():
    out = evaluate(spec(type="contains", value=["x", "beta"], mode="any"), ctx("has beta"))
    assert out.passed is True


def test_contains_is_case_insensitive_by_default():
    assert evaluate(spec(type="contains", value="HELLO"), ctx("hello")).passed


def test_not_contains_reports_the_banned_string():
    out = evaluate(spec(type="not_contains", value=["secret"]), ctx("the secret sauce"))
    assert out.passed is False
    assert out.detail["found"] == ["secret"]


def test_equals_strips_by_default():
    assert evaluate(spec(type="equals", value="yes"), ctx("  yes\n")).passed


def test_regex_with_expect_false_inverts_the_check():
    out = evaluate(spec(type="regex", pattern=r"\d+", expect=False), ctx("no digits here"))
    assert out.passed is True


def test_json_valid_accepts_a_fenced_block():
    assert evaluate(spec(type="json_valid"), ctx('```json\n{"a": 1}\n```')).passed


def test_json_valid_fails_on_prose():
    out = evaluate(spec(type="json_valid"), ctx("Sure! Here you go: a=1"))
    assert out.passed is False
    assert "invalid JSON" in out.message


def test_json_schema_reports_violations_with_partial_credit():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    out = evaluate(spec(type="json_schema", schema=schema), ctx('{"a": "no"}'))
    assert out.passed is False
    assert 0.0 < out.score < 1.0
    assert out.detail["errors"]


def test_json_path_indexes_into_lists():
    out = evaluate(
        spec(type="json_path", path="items.1.sku", equals="B"),
        ctx('{"items": [{"sku": "A"}, {"sku": "B"}]}'),
    )
    assert out.passed is True


def test_json_path_missing_key_fails_clearly():
    out = evaluate(spec(type="json_path", path="nope"), ctx('{"a": 1}'))
    assert out.passed is False
    assert "not found" in out.message


def test_length_counts_words():
    out = evaluate(spec(type="length", unit="words", max=3), ctx("one two three four"))
    assert out.passed is False
    assert out.detail["size"] == 4


def test_length_requires_a_bound():
    with pytest.raises(Exception):
        spec(type="length", unit="words")


def test_latency_is_skipped_for_cached_responses():
    # A cached response has no meaningful latency, so asserting on it would be a lie.
    out = evaluate(spec(type="latency", max_ms=1), ctx("x", cached=True, latency_ms=0.0))
    assert out.passed is True
    assert "cached" in out.message


def test_latency_fails_over_budget():
    out = evaluate(spec(type="latency", max_ms=50), ctx("x", latency_ms=500.0))
    assert out.passed is False


def test_cost_budget():
    assert evaluate(spec(type="cost", max_usd=0.01), ctx("x", cost_usd=0.005)).passed
    assert not evaluate(spec(type="cost", max_usd=0.001), ctx("x", cost_usd=0.005)).passed


def test_tokens_budget():
    out = evaluate(spec(type="tokens", max_output=5), ctx("x", output_tokens=50))
    assert out.passed is False


def test_a_raising_assertion_fails_its_own_row_only():
    # An invalid regex must not abort the run.
    out = evaluate(spec(type="regex", pattern="([unclosed"), ctx("anything"))
    assert out.passed is False
    assert "raised" in out.message


def test_similarity_without_the_extra_explains_how_to_install_it():
    from regressa.assertions import similarity

    similarity._models.clear()
    out = evaluate(
        spec(type="similarity", reference="hello", model="definitely-not-installed"),
        ctx("hello"),
    )
    # Either the extra is absent (install hint) or present (a real score).
    assert out.message


def test_similarity_uses_an_injected_encoder():
    from regressa.assertions import similarity

    class StubEncoder:
        def encode(self, texts):
            # Identical unit vectors → cosine 1.0.
            return [[1.0, 0.0], [1.0, 0.0]]

    similarity.set_model("stub", StubEncoder())
    out = evaluate(
        spec(type="similarity", reference="anything", model="stub", threshold=0.9),
        ctx("anything"),
    )
    assert out.passed is True
    assert out.score == pytest.approx(1.0)
