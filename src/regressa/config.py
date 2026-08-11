"""Suite configuration schema.

A suite is a YAML file describing three things:

    targets:   which model/prompt configurations to exercise
    cases:     the fixed prompts to send
    assertions: what a correct output looks like

Everything is validated with Pydantic so a typo in a suite is a startup error
with a line-level message, not a confusing mid-run failure.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from regressa import capabilities


class _Strict(BaseModel):
    """Reject unknown keys — a misspelled field should never be silently ignored."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #


class AssertionBase(_Strict):
    label: str | None = Field(
        default=None, description="Optional human name shown in reports."
    )
    weight: float = Field(
        default=1.0, gt=0, description="Relative weight in the case's score."
    )


class ContainsAssertion(AssertionBase):
    type: Literal["contains"]
    value: str | list[str]
    mode: Literal["all", "any"] = "all"
    ignore_case: bool = True


class NotContainsAssertion(AssertionBase):
    type: Literal["not_contains"]
    value: str | list[str]
    ignore_case: bool = True


class EqualsAssertion(AssertionBase):
    type: Literal["equals"]
    value: str
    ignore_case: bool = False
    strip: bool = True


class RegexAssertion(AssertionBase):
    type: Literal["regex"]
    pattern: str
    ignore_case: bool = False
    dotall: bool = False
    expect: bool = Field(default=True, description="False asserts the pattern must NOT match.")


class StartsWithAssertion(AssertionBase):
    type: Literal["starts_with"]
    value: str
    ignore_case: bool = True


class EndsWithAssertion(AssertionBase):
    type: Literal["ends_with"]
    value: str
    ignore_case: bool = True


class JsonValidAssertion(AssertionBase):
    type: Literal["json_valid"]


class JsonSchemaAssertion(AssertionBase):
    type: Literal["json_schema"]
    schema_: dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class JsonPathAssertion(AssertionBase):
    """Dotted-path lookup into JSON output, e.g. `order.items.0.sku`."""

    type: Literal["json_path"]
    path: str
    equals: Any = None
    exists: bool = True

    @model_validator(mode="after")
    def _check(self) -> JsonPathAssertion:
        if self.equals is not None and not self.exists:
            raise ValueError("json_path: `equals` is meaningless with `exists: false`")
        return self


class LengthAssertion(AssertionBase):
    type: Literal["length"]
    # `sentences` exists because "answer in N sentences or fewer" is one of the
    # most common instructions in a real system prompt, and a word count is a
    # poor proxy for it.
    unit: Literal["chars", "words", "lines", "sentences"] = "chars"
    min: int | None = None
    max: int | None = None

    @model_validator(mode="after")
    def _check(self) -> LengthAssertion:
        if self.min is None and self.max is None:
            raise ValueError("length: set at least one of `min` / `max`")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("length: `min` must not exceed `max`")
        return self


class LatencyAssertion(AssertionBase):
    type: Literal["latency"]
    max_ms: float = Field(gt=0)


class CostAssertion(AssertionBase):
    type: Literal["cost"]
    max_usd: float = Field(gt=0)


class TokensAssertion(AssertionBase):
    type: Literal["tokens"]
    max_output: int | None = Field(default=None, gt=0)
    max_input: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> TokensAssertion:
        if self.max_output is None and self.max_input is None:
            raise ValueError("tokens: set at least one of `max_output` / `max_input`")
        return self


class LlmJudgeAssertion(AssertionBase):
    """Grade an open-ended output against a rubric using a grader model.

    Costs money and is not fully deterministic — keep it for prose quality and
    lean on the deterministic assertions for the regression backbone.
    """

    type: Literal["llm_judge"]
    rubric: str
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    model: str = "claude-opus-5"
    max_tokens: int = Field(default=1024, gt=0)


class SimilarityAssertion(AssertionBase):
    """Embedding cosine similarity against a reference answer.

    Requires the `similarity` extra (`pip install 'regressa[similarity]'`) because
    the Anthropic API has no embeddings endpoint — this runs a local model.
    """

    type: Literal["similarity"]
    reference: str
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


AssertionSpec = Annotated[
    Union[
        ContainsAssertion,
        NotContainsAssertion,
        EqualsAssertion,
        RegexAssertion,
        StartsWithAssertion,
        EndsWithAssertion,
        JsonValidAssertion,
        JsonSchemaAssertion,
        JsonPathAssertion,
        LengthAssertion,
        LatencyAssertion,
        CostAssertion,
        TokensAssertion,
        LlmJudgeAssertion,
        SimilarityAssertion,
    ],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


class Target(_Strict):
    """One model configuration under test.

    A "target" is the thing whose change you are hunting for regressions in:
    swap the model, edit the system prompt, or bump effort, and re-run.
    """

    id: str
    provider: Literal["anthropic", "mock"] = "anthropic"
    model: str
    system: str | None = None
    max_tokens: int = Field(default=1024, gt=0)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    thinking: Literal["adaptive", "disabled"] | None = None
    stop_sequences: list[str] | None = None
    # Only sent when the model accepts it; see regressa.capabilities.
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    # Provider-specific extras (e.g. mock responses). Passed through untouched.
    options: dict[str, Any] = Field(default_factory=dict)

    def validate_against_model(self) -> list[capabilities.ValidationIssue]:
        return capabilities.validate_target(
            self.id,
            self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            effort=self.effort,
            thinking=self.thinking,
            max_tokens=self.max_tokens,
        )


# --------------------------------------------------------------------------- #
# Cases and suites
# --------------------------------------------------------------------------- #


class Case(_Strict):
    id: str
    prompt: str
    description: str | None = None
    vars: dict[str, Any] = Field(default_factory=dict)
    assertions: list[AssertionSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Restrict this case to a subset of the suite's targets.
    targets: list[str] | None = None

    @field_validator("id")
    @classmethod
    def _no_blank_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("case id must not be blank")
        return v


class Defaults(_Strict):
    """Values applied to every case/target unless overridden locally."""

    max_tokens: int | None = Field(default=None, gt=0)
    system: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    thinking: Literal["adaptive", "disabled"] | None = None
    # Appended to every case's own assertion list — good for global guardrails
    # like "never leak the system prompt".
    assertions: list[AssertionSpec] = Field(default_factory=list)


class Suite(_Strict):
    name: str
    description: str | None = None
    vars: dict[str, Any] = Field(default_factory=dict)
    defaults: Defaults = Field(default_factory=Defaults)
    targets: list[Target]
    cases: list[Case]

    @model_validator(mode="after")
    def _check_ids(self) -> Suite:
        target_ids = [t.id for t in self.targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("duplicate target ids in suite")
        case_ids = [c.id for c in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("duplicate case ids in suite")
        known = set(target_ids)
        for case in self.cases:
            for ref in case.targets or []:
                if ref not in known:
                    raise ValueError(
                        f"case {case.id!r} references unknown target {ref!r}"
                    )
        return self

    def resolved_targets(self, case: Case) -> list[Target]:
        """Targets this case runs against, with suite defaults folded in."""
        selected = (
            [t for t in self.targets if t.id in set(case.targets)]
            if case.targets
            else list(self.targets)
        )
        out: list[Target] = []
        for target in selected:
            merged = target.model_copy()
            if merged.system is None:
                merged.system = self.defaults.system
            if self.defaults.max_tokens is not None and "max_tokens" not in (
                target.model_fields_set
            ):
                merged.max_tokens = self.defaults.max_tokens
            if merged.effort is None:
                merged.effort = self.defaults.effort
            if merged.thinking is None:
                merged.thinking = self.defaults.thinking
            out.append(merged)
        return out

    def resolved_assertions(self, case: Case) -> list[Any]:
        return [*case.assertions, *self.defaults.assertions]

    def fingerprint(self) -> str:
        """Stable hash of the suite definition.

        Recorded on every run so `regressa diff` can warn when two runs were
        produced from different suite definitions — a score change then may not
        be a regression at all.
        """
        payload = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def validate_targets(self) -> list[capabilities.ValidationIssue]:
        issues: list[capabilities.ValidationIssue] = []
        for target in self.targets:
            if target.provider == "mock":
                continue
            issues.extend(target.validate_against_model())
        return issues
