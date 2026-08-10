"""Assertion registry.

Importing the submodules is what registers their evaluators, so this module must
stay imported for `evaluate()` to resolve anything.
"""

from __future__ import annotations

from regressa.assertions import deterministic, judge, similarity  # noqa: F401
from regressa.assertions.base import (
    EvalContext,
    evaluate,
    evaluator,
    outcome,
    registered_types,
)

__all__ = [
    "EvalContext",
    "evaluate",
    "evaluator",
    "outcome",
    "registered_types",
]
