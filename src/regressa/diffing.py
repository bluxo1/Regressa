"""Run-to-run comparison — the part that actually flags regressions.

A regression is defined narrowly on purpose:

  * a case that passed and now fails, or
  * a case whose score dropped by more than a tolerance.

Deliberately *not* regressions: a case that errored (provider/network failure —
that's a harness problem, reported separately), and a newly added case that
fails (it never passed, so nothing regressed). Blurring these makes the gate
noisy, and a noisy gate gets ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from regressa.results import CaseResult, RunSummary

# Score moves smaller than this are treated as noise. Judge assertions in
# particular jitter slightly between identical runs.
DEFAULT_TOLERANCE = 0.05


class Change(str, Enum):
    REGRESSED = "regressed"
    FIXED = "fixed"
    SCORE_DROP = "score_drop"
    SCORE_GAIN = "score_gain"
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    ERRORED = "errored"


@dataclass
class CaseDelta:
    case_id: str
    target_id: str
    change: Change
    before_passed: bool | None = None
    after_passed: bool | None = None
    before_score: float | None = None
    after_score: float | None = None
    detail: str = ""

    @property
    def score_delta(self) -> float | None:
        if self.before_score is None or self.after_score is None:
            return None
        return self.after_score - self.before_score

    @property
    def is_regression(self) -> bool:
        return self.change in (Change.REGRESSED, Change.SCORE_DROP)


@dataclass
class RunDiff:
    before: RunSummary
    after: RunSummary
    tolerance: float = DEFAULT_TOLERANCE
    deltas: list[CaseDelta] = field(default_factory=list)
    # True when the two runs came from different suite definitions, in which case
    # differences may reflect an edited suite rather than a real regression.
    suite_changed: bool = False

    def by_change(self, change: Change) -> list[CaseDelta]:
        return [d for d in self.deltas if d.change is change]

    @property
    def regressions(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.is_regression]

    @property
    def fixes(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.change in (Change.FIXED, Change.SCORE_GAIN)]

    @property
    def errors(self) -> list[CaseDelta]:
        return self.by_change(Change.ERRORED)

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    @property
    def pass_rate_delta(self) -> float:
        return self.after.pass_rate - self.before.pass_rate

    @property
    def score_delta(self) -> float:
        return self.after.score - self.before.score

    @property
    def cost_delta(self) -> float:
        return self.after.cost_usd - self.before.cost_usd


def diff_runs(
    before: RunSummary,
    after: RunSummary,
    tolerance: float = DEFAULT_TOLERANCE,
) -> RunDiff:
    result = RunDiff(
        before=before,
        after=after,
        tolerance=tolerance,
        suite_changed=before.suite_hash != after.suite_hash,
    )

    before_map = {r.key: r for r in before.results}
    after_map = {r.key: r for r in after.results}

    for key in sorted(before_map.keys() | after_map.keys()):
        case_id, target_id = key
        old = before_map.get(key)
        new = after_map.get(key)

        if old is None and new is not None:
            result.deltas.append(
                CaseDelta(
                    case_id=case_id,
                    target_id=target_id,
                    change=Change.ADDED,
                    after_passed=new.passed,
                    after_score=new.score,
                    detail="new case/target pair; not compared",
                )
            )
            continue

        if new is None and old is not None:
            result.deltas.append(
                CaseDelta(
                    case_id=case_id,
                    target_id=target_id,
                    change=Change.REMOVED,
                    before_passed=old.passed,
                    before_score=old.score,
                    detail="present in baseline but not in this run",
                )
            )
            continue

        assert old is not None and new is not None
        result.deltas.append(_compare(old, new, tolerance))

    return result


def _compare(old: CaseResult, new: CaseResult, tolerance: float) -> CaseDelta:
    base = {
        "case_id": new.case_id,
        "target_id": new.target_id,
        "before_passed": old.passed,
        "after_passed": new.passed,
        "before_score": old.score,
        "after_score": new.score,
    }

    if new.errored:
        return CaseDelta(
            **base,
            change=Change.ERRORED,
            detail=f"harness error, not a regression: {new.error}",
        )

    if old.passed and not new.passed:
        return CaseDelta(**base, change=Change.REGRESSED, detail=_why(new))

    if not old.passed and new.passed:
        return CaseDelta(**base, change=Change.FIXED, detail="now passing")

    delta = new.score - old.score
    if delta < -tolerance:
        return CaseDelta(
            **base,
            change=Change.SCORE_DROP,
            detail=f"score fell {old.score:.3f} -> {new.score:.3f} ({delta:+.3f})",
        )
    if delta > tolerance:
        return CaseDelta(
            **base,
            change=Change.SCORE_GAIN,
            detail=f"score rose {old.score:.3f} -> {new.score:.3f} ({delta:+.3f})",
        )

    return CaseDelta(**base, change=Change.UNCHANGED)


def _why(result: CaseResult) -> str:
    """Name the assertions responsible for a newly failing case."""
    failed = [a for a in result.assertions if not a.passed]
    if not failed:
        return "now failing"
    return "; ".join(f"{a.label or a.type}: {a.message}" for a in failed[:3])
