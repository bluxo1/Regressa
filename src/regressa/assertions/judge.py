"""LLM-as-judge assertion.

For outputs whose correctness can't be pinned down with a string or schema check
— tone, completeness, whether an explanation is actually correct. It costs money
and is not perfectly deterministic, so treat a judge score as a quality signal
and keep hard pass/fail gates on the deterministic assertions where you can.

The judge is asked for structured output, so its verdict is parsed from a
schema-constrained JSON object rather than scraped out of prose.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from regressa import capabilities
from regressa.assertions.base import EvalContext, evaluator, outcome

JUDGE_SYSTEM = """You are a strict evaluator in an automated test suite.

Score how well the CANDIDATE OUTPUT satisfies the RUBRIC, on a 0.0-1.0 scale:
  1.0  fully satisfies every part of the rubric
  0.5  satisfies the rubric partially, or satisfies it with a material flaw
  0.0  does not satisfy the rubric

Judge only against the rubric. Do not reward style, length, or confidence that
the rubric does not ask for, and do not penalise wording differences that leave
the rubric satisfied. Be consistent: the same output and rubric must always get
the same score, because this score is compared across runs to detect regressions.

Give one sentence of reasoning naming the specific rubric element that decided
the score."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "0.0 to 1.0"},
        "reasoning": {"type": "string", "description": "One sentence."},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}

_lock = threading.Lock()
_client: Any = None


def _get_client() -> Any:
    global _client
    with _lock:
        if _client is None:
            import anthropic

            _client = anthropic.Anthropic()
        return _client


def set_client(client: Any) -> None:
    """Inject a client (used by the test suite to avoid network calls)."""
    global _client
    with _lock:
        _client = client


def _build_prompt(rubric: str, ctx: EvalContext) -> str:
    return (
        f"<rubric>\n{rubric}\n</rubric>\n\n"
        f"<prompt_given_to_model>\n{ctx.prompt}\n</prompt_given_to_model>\n\n"
        f"<candidate_output>\n{ctx.output}\n</candidate_output>"
    )


@evaluator("llm_judge")
def _llm_judge(spec: Any, ctx: EvalContext):
    if not ctx.output.strip():
        return outcome(spec, passed=False, score=0.0, message="empty output; nothing to judge")

    client = _get_client()
    message = client.messages.create(
        model=spec.model,
        max_tokens=spec.max_tokens,
        system=JUDGE_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": _build_prompt(spec.rubric, ctx)}],
    )

    if getattr(message, "stop_reason", None) == "refusal":
        return outcome(spec, passed=False, score=0.0,
                       message="grader declined to score this output")

    verdict = _parse(message)
    if verdict is None:
        return outcome(spec, passed=False, score=0.0,
                       message="grader returned unparseable output")

    score = max(0.0, min(1.0, float(verdict.get("score", 0.0))))
    reasoning = str(verdict.get("reasoning", "")).strip()
    judge_cost = _judge_cost(spec.model, message)

    return outcome(
        spec,
        passed=score >= spec.threshold,
        score=score,
        message=f"judge {score:.2f} (threshold {spec.threshold:.2f}): {reasoning}",
        # The runner folds judge_cost_usd into the case's total cost so grading
        # spend shows up in reports rather than hiding off-ledger.
        detail={"score": score, "reasoning": reasoning, "judge_cost_usd": judge_cost},
    )


def _parse(message: Any) -> dict[str, Any] | None:
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        try:
            parsed = json.loads(getattr(block, "text", ""))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _judge_cost(model: str, message: Any) -> float:
    usage = getattr(message, "usage", None)
    if usage is None:
        return 0.0
    try:
        return capabilities.price(
            model,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
            int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )
    except capabilities.UnknownModelError:
        return 0.0
