"""Runtime result objects produced by a Regressa run.

These are deliberately separate from `regressa.config` (which describes *inputs*).
Everything here is serialisable so it can be persisted to SQLite and diffed later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AssertionOutcome(BaseModel):
    """Result of evaluating one assertion against one model output."""

    type: str
    label: str | None = None
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    weight: float = 1.0
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


class CaseResult(BaseModel):
    """Result of running one case against one target."""

    case_id: str
    target_id: str
    passed: bool
    score: float = 0.0
    output: str = ""
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    cached: bool = False
    stop_reason: str | None = None
    error: str | None = None
    assertions: list[AssertionOutcome] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.case_id, self.target_id)

    @property
    def errored(self) -> bool:
        return self.error is not None


class RunSummary(BaseModel):
    """Aggregate view of a single `regressa run` invocation."""

    run_id: str
    suite: str
    suite_hash: str
    label: str | None = None
    git_sha: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    score: float = 0.0
    cost_usd: float = 0.0
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def recompute(self) -> RunSummary:
        self.total = len(self.results)
        self.passed = sum(1 for r in self.results if r.passed)
        self.errors = sum(1 for r in self.results if r.errored)
        self.failed = self.total - self.passed
        self.score = (
            sum(r.score for r in self.results) / self.total if self.total else 0.0
        )
        self.cost_usd = sum(r.usage.cost_usd for r in self.results)
        return self
