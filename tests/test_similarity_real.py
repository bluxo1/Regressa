"""Semantic-similarity assertion against a real embedding model.

Skipped unless the `similarity` extra is installed, so the default test run stays
fast and dependency-light:

    pip install -e '.[similarity]'
    pytest tests/test_similarity_real.py

Everything else about the assertion is covered by stubbed tests in
test_assertions.py. What only a real model can verify is that the library loads,
that `encode` returns something the cosine helper can consume, and that the
scores actually rank paraphrases above unrelated text — which is the entire
premise of the assertion.

The model downloads on first use (~90MB) and is cached by
sentence-transformers afterwards.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from regressa.assertions import EvalContext, evaluate
from regressa.config import AssertionSpec
from regressa.providers.base import CompletionResponse

pytest.importorskip(
    "sentence_transformers",
    reason="needs the `similarity` extra: pip install -e '.[similarity]'",
)

pytestmark = pytest.mark.slow

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_adapter = TypeAdapter(AssertionSpec)


def spec(**kwargs):
    kwargs.setdefault("type", "similarity")
    kwargs.setdefault("model", MODEL)
    return _adapter.validate_python(kwargs)


def ctx(output: str) -> EvalContext:
    return EvalContext(
        output=output,
        response=CompletionResponse(text=output),
        cost_usd=0.0,
        case_id="c",
        target_id="t",
        prompt="p",
        vars={},
    )


REFERENCE = "Refunds are available within 30 days of purchase."


@pytest.fixture(scope="module", autouse=True)
def _warm_model():
    """Load the model once so the first test isn't timed against a download."""
    from regressa.assertions.similarity import _load

    _load(MODEL)


def test_identical_text_scores_essentially_one():
    out = evaluate(spec(reference=REFERENCE, threshold=0.9), ctx(REFERENCE))
    assert out.passed is True
    assert out.score == pytest.approx(1.0, abs=1e-3)


def test_a_paraphrase_scores_high():
    out = evaluate(
        spec(reference=REFERENCE, threshold=0.6),
        ctx("You can get your money back up to 30 days after buying."),
    )
    assert out.passed is True, out.message
    assert out.score > 0.6


def test_unrelated_text_scores_low_and_fails():
    out = evaluate(
        spec(reference=REFERENCE, threshold=0.6),
        ctx("The Mars rover collected several rock samples this week."),
    )
    assert out.passed is False
    assert out.score < 0.4, out.message


def test_scores_rank_paraphrase_above_unrelated():
    """The ordering is the property the assertion actually depends on."""
    paraphrase = evaluate(
        spec(reference=REFERENCE, threshold=0.0),
        ctx("Purchases can be refunded for a month."),
    ).score
    unrelated = evaluate(
        spec(reference=REFERENCE, threshold=0.0),
        ctx("Please restart your router and try again."),
    ).score
    assert paraphrase > unrelated


def test_a_contradiction_still_scores_high_which_is_the_known_limitation():
    """Embedding similarity measures topic, not truth.

    "within 30 days" and "no refunds ever" are about the same subject and score
    similarly, so `similarity` must not be used to assert policy correctness —
    that is what `contains` and `llm_judge` are for. Pinning the behaviour here
    so the limitation is documented in something that runs.
    """
    out = evaluate(
        spec(reference=REFERENCE, threshold=0.0),
        ctx("We never offer refunds under any circumstances."),
    )
    assert out.score > 0.5, out.message


def test_score_is_clamped_into_the_unit_interval():
    out = evaluate(spec(reference=REFERENCE, threshold=0.0), ctx("Unrelated nonsense."))
    assert 0.0 <= out.score <= 1.0


def test_detail_records_the_embedding_model_used():
    out = evaluate(spec(reference=REFERENCE, threshold=0.0), ctx(REFERENCE))
    assert out.detail["embedding_model"] == MODEL
    assert "cosine" in out.detail
