"""Assertion evaluation context and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from regressa.providers.base import CompletionResponse
from regressa.results import AssertionOutcome


@dataclass
class EvalContext:
    """Everything an assertion may inspect about a single case/target run."""

    output: str
    response: CompletionResponse
    cost_usd: float
    case_id: str
    target_id: str
    prompt: str
    vars: dict[str, Any]


Evaluator = Callable[[Any, EvalContext], AssertionOutcome]

_EVALUATORS: dict[str, Evaluator] = {}


def evaluator(type_name: str) -> Callable[[Evaluator], Evaluator]:
    def decorate(fn: Evaluator) -> Evaluator:
        _EVALUATORS[type_name] = fn
        return fn

    return decorate


def evaluate(spec: Any, ctx: EvalContext) -> AssertionOutcome:
    """Run one assertion, converting unexpected errors into a failed outcome.

    An assertion that raises would otherwise abort the whole run; here it fails
    loudly on its own row instead, so one bad regex doesn't hide 200 good results.
    """
    fn = _EVALUATORS.get(spec.type)
    if fn is None:
        return AssertionOutcome(
            type=spec.type,
            label=getattr(spec, "label", None),
            passed=False,
            score=0.0,
            weight=getattr(spec, "weight", 1.0),
            message=f"no evaluator registered for assertion type {spec.type!r}",
        )
    try:
        return fn(spec, ctx)
    except Exception as exc:  # noqa: BLE001 - reported as a failing assertion
        return AssertionOutcome(
            type=spec.type,
            label=getattr(spec, "label", None),
            passed=False,
            score=0.0,
            weight=getattr(spec, "weight", 1.0),
            message=f"assertion raised {type(exc).__name__}: {exc}",
        )


def outcome(
    spec: Any,
    *,
    passed: bool,
    message: str = "",
    score: float | None = None,
    detail: dict[str, Any] | None = None,
) -> AssertionOutcome:
    """Build an outcome, defaulting the score to the binary pass/fail value."""
    return AssertionOutcome(
        type=spec.type,
        label=getattr(spec, "label", None),
        passed=passed,
        score=(1.0 if passed else 0.0) if score is None else max(0.0, min(1.0, score)),
        weight=getattr(spec, "weight", 1.0),
        message=message,
        detail=detail or {},
    )


def registered_types() -> list[str]:
    return sorted(_EVALUATORS)
