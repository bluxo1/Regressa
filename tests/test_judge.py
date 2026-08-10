"""LLM-as-judge assertion, exercised with a stub grader client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import TypeAdapter

from regressa.assertions import EvalContext, evaluate, judge
from regressa.config import AssertionSpec, Suite
from regressa.providers.base import CompletionResponse
from regressa.runner import Runner

_adapter = TypeAdapter(AssertionSpec)


@dataclass
class Block:
    type: str
    text: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Message:
    content: list[Block] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = "end_turn"


class StubGrader:
    """Stands in for the Anthropic client used by the judge."""

    def __init__(self, message: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._message = message

        class Messages:
            def __init__(self, outer: StubGrader) -> None:
                self._outer = outer

            def create(self, **params: Any) -> Any:
                self._outer.calls.append(params)
                return self._outer._message

        self.messages = Messages(self)


def verdict(score: float, reasoning: str = "because", **usage: int) -> Message:
    return Message(
        content=[Block(type="text", text=json.dumps({"score": score, "reasoning": reasoning}))],
        usage=Usage(**usage),
    )


def spec(**kwargs: Any):
    kwargs.setdefault("type", "llm_judge")
    kwargs.setdefault("rubric", "The answer must be correct.")
    return _adapter.validate_python(kwargs)


def ctx(output: str = "some answer") -> EvalContext:
    return EvalContext(
        output=output,
        response=CompletionResponse(text=output),
        cost_usd=0.0,
        case_id="c",
        target_id="t",
        prompt="the original prompt",
        vars={},
    )


@pytest.fixture(autouse=True)
def _reset_client():
    yield
    judge.set_client(None)


def test_score_above_threshold_passes():
    judge.set_client(StubGrader(verdict(0.9)))
    out = evaluate(spec(threshold=0.7), ctx())
    assert out.passed is True
    assert out.score == pytest.approx(0.9)
    assert "because" in out.message


def test_score_below_threshold_fails():
    judge.set_client(StubGrader(verdict(0.4)))
    out = evaluate(spec(threshold=0.7), ctx())
    assert out.passed is False
    assert out.score == pytest.approx(0.4)


def test_scores_are_clamped_to_the_unit_interval():
    judge.set_client(StubGrader(verdict(4.2)))
    assert evaluate(spec(), ctx()).score == pytest.approx(1.0)


def test_unparseable_grader_output_fails_rather_than_passing_silently():
    judge.set_client(StubGrader(Message(content=[Block(type="text", text="looks fine to me")])))
    out = evaluate(spec(), ctx())
    assert out.passed is False
    assert "unparseable" in out.message


def test_a_grader_refusal_fails_the_assertion():
    judge.set_client(StubGrader(Message(content=[], stop_reason="refusal")))
    out = evaluate(spec(), ctx())
    assert out.passed is False
    assert "declined" in out.message


def test_empty_output_is_not_sent_to_the_grader():
    grader = StubGrader(verdict(1.0))
    judge.set_client(grader)
    out = evaluate(spec(), ctx(output="   "))
    assert out.passed is False
    assert grader.calls == []  # no spend on an empty candidate


def test_the_rubric_and_prompt_are_both_given_to_the_grader():
    grader = StubGrader(verdict(1.0))
    judge.set_client(grader)
    evaluate(spec(rubric="MY RUBRIC"), ctx(output="MY OUTPUT"))
    sent = grader.calls[0]["messages"][0]["content"]
    assert "MY RUBRIC" in sent
    assert "MY OUTPUT" in sent
    assert "the original prompt" in sent


def test_the_grader_is_asked_for_structured_output():
    grader = StubGrader(verdict(1.0))
    judge.set_client(grader)
    evaluate(spec(), ctx())
    fmt = grader.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "score" in fmt["schema"]["properties"]


def test_grader_never_receives_sampling_parameters():
    # The default judge model rejects temperature with a 400.
    grader = StubGrader(verdict(1.0))
    judge.set_client(grader)
    evaluate(spec(), ctx())
    assert "temperature" not in grader.calls[0]
    assert "top_p" not in grader.calls[0]


def test_grading_cost_is_computed_from_grader_usage():
    judge.set_client(StubGrader(verdict(1.0, input_tokens=1_000_000, output_tokens=0)))
    out = evaluate(spec(model="claude-opus-5"), ctx())
    assert out.detail["judge_cost_usd"] == pytest.approx(5.0)


def test_grading_cost_lands_on_the_case_total(store):
    """Grader spend must appear on the ledger, not hide inside the assertion."""
    judge.set_client(StubGrader(verdict(1.0, input_tokens=1_000_000, output_tokens=0)))
    suite = Suite.model_validate(
        {
            "name": "judged",
            "targets": [
                {"id": "m", "provider": "mock", "model": "mock-1",
                 "options": {"default": "an answer"}}
            ],
            "cases": [
                {
                    "id": "c",
                    "prompt": "p",
                    "assertions": [
                        {"type": "llm_judge", "rubric": "r", "model": "claude-opus-5"}
                    ],
                }
            ],
        }
    )
    run = Runner(store).run(suite)
    # The mock target itself is free, so the whole cost is the grader's.
    assert run.cost_usd == pytest.approx(5.0)
