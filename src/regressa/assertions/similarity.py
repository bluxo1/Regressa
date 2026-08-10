"""Embedding-similarity assertion.

Note on the design: the Anthropic API has no embeddings endpoint, so this cannot
be served by the same client as everything else. It runs a local
`sentence-transformers` model instead, which is why it lives behind an optional
extra:

    pip install 'regressa[similarity]'

Without that extra, a suite using `similarity` fails with a clear install
message rather than silently passing.
"""

from __future__ import annotations

import threading
from typing import Any

from regressa.assertions.base import EvalContext, evaluator, outcome

_lock = threading.Lock()
_models: dict[str, Any] = {}

INSTALL_HINT = (
    "the `similarity` assertion needs a local embedding model. Install it with "
    "`pip install 'regressa[similarity]'`. (The Anthropic API has no embeddings "
    "endpoint, so this cannot use your API key.)"
)


class SimilarityUnavailable(RuntimeError):
    pass


def _load(model_name: str) -> Any:
    with _lock:
        if model_name not in _models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise SimilarityUnavailable(INSTALL_HINT) from exc
            _models[model_name] = SentenceTransformer(model_name)
        return _models[model_name]


def set_model(model_name: str, model: Any) -> None:
    """Inject a preloaded/stub encoder (used by the test suite)."""
    with _lock:
        _models[model_name] = model


@evaluator("similarity")
def _similarity(spec: Any, ctx: EvalContext):
    if not ctx.output.strip():
        return outcome(spec, passed=False, score=0.0, message="empty output")

    try:
        model = _load(spec.model)
    except SimilarityUnavailable as exc:
        return outcome(spec, passed=False, score=0.0, message=str(exc))

    vectors = model.encode([ctx.output, spec.reference])
    score = _cosine(vectors[0], vectors[1])
    # Cosine similarity is in [-1, 1]; clamp so the score stays a 0-1 quality signal.
    clamped = max(0.0, min(1.0, float(score)))

    return outcome(
        spec,
        passed=clamped >= spec.threshold,
        score=clamped,
        message=f"cosine {clamped:.3f} (threshold {spec.threshold:.2f})",
        detail={"cosine": clamped, "embedding_model": spec.model},
    )


def _cosine(a: Any, b: Any) -> float:
    # Implemented longhand so numpy stays an implicit dependency of the extra
    # rather than a hard requirement of the core install.
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    norm_a = sum(float(x) * float(x) for x in a) ** 0.5
    norm_b = sum(float(y) * float(y) for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
