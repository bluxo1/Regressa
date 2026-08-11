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


# --------------------------------------------------------------------------- #
# Target-vs-target comparison (A/B two prompt versions within one run)
# --------------------------------------------------------------------------- #
#
# This answers a different question from `diff_runs`. That one asks "did today
# break what worked yesterday" — same target, two points in time. This asks "is
# version B better than version A" — two targets, same inputs, same moment, so
# model drift and flake cancel out.


class Verdict(str, Enum):
    A_ONLY = "a_only"      # A passes, B fails — B is a step back
    B_ONLY = "b_only"      # B passes, A fails — B is an improvement
    BOTH_PASS = "both_pass"
    BOTH_FAIL = "both_fail"
    A_MISSING = "a_missing"
    B_MISSING = "b_missing"


@dataclass
class TargetDelta:
    case_id: str
    verdict: Verdict
    a_passed: bool | None = None
    b_passed: bool | None = None
    a_score: float | None = None
    b_score: float | None = None
    detail: str = ""

    @property
    def score_delta(self) -> float | None:
        if self.a_score is None or self.b_score is None:
            return None
        return self.b_score - self.a_score


@dataclass
class TargetComparison:
    run_id: str
    target_a: str
    target_b: str
    deltas: list[TargetDelta] = field(default_factory=list)

    def by_verdict(self, verdict: Verdict) -> list[TargetDelta]:
        return [d for d in self.deltas if d.verdict is verdict]

    @property
    def a_wins(self) -> list[TargetDelta]:
        return self.by_verdict(Verdict.A_ONLY)

    @property
    def b_wins(self) -> list[TargetDelta]:
        return self.by_verdict(Verdict.B_ONLY)

    @property
    def b_is_worse(self) -> bool:
        """True when B fails something A passed — the gate for promoting B."""
        return bool(self.a_wins)

    @property
    def a_score(self) -> float:
        scores = [d.a_score for d in self.deltas if d.a_score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def b_score(self) -> float:
        scores = [d.b_score for d in self.deltas if d.b_score is not None]
        return sum(scores) / len(scores) if scores else 0.0


def compare_targets(run: RunSummary, target_a: str, target_b: str) -> TargetComparison:
    """A/B two targets inside a single run.

    Raises ValueError when a target id isn't present, rather than silently
    reporting every case as missing — a typo in a target id should be loud.
    """
    present = {r.target_id for r in run.results}
    for name in (target_a, target_b):
        if name not in present:
            raise ValueError(
                f"target {name!r} is not in run {run.run_id}. "
                f"Targets in this run: {', '.join(sorted(present)) or '(none)'}"
            )

    a_map = {r.case_id: r for r in run.results if r.target_id == target_a}
    b_map = {r.case_id: r for r in run.results if r.target_id == target_b}

    comparison = TargetComparison(run_id=run.run_id, target_a=target_a, target_b=target_b)

    for case_id in sorted(a_map.keys() | b_map.keys()):
        a = a_map.get(case_id)
        b = b_map.get(case_id)

        if a is None:
            comparison.deltas.append(
                TargetDelta(
                    case_id=case_id,
                    verdict=Verdict.A_MISSING,
                    b_passed=b.passed if b else None,
                    b_score=b.score if b else None,
                    detail=f"case not run against {target_a}",
                )
            )
            continue
        if b is None:
            comparison.deltas.append(
                TargetDelta(
                    case_id=case_id,
                    verdict=Verdict.B_MISSING,
                    a_passed=a.passed,
                    a_score=a.score,
                    detail=f"case not run against {target_b}",
                )
            )
            continue

        # Name the target in the detail: "only B passes" plus a bare failure
        # reason is ambiguous about whose failure is being described.
        if a.passed and not b.passed:
            verdict, detail = Verdict.A_ONLY, f"{target_b} failed - {_why(b)}"
        elif b.passed and not a.passed:
            verdict, detail = Verdict.B_ONLY, f"{target_a} failed - {_why(a)}"
        elif a.passed and b.passed:
            verdict, detail = Verdict.BOTH_PASS, ""
        else:
            verdict, detail = Verdict.BOTH_FAIL, f"{target_b} failed - {_why(b)}"

        comparison.deltas.append(
            TargetDelta(
                case_id=case_id,
                verdict=verdict,
                a_passed=a.passed,
                b_passed=b.passed,
                a_score=a.score,
                b_score=b.score,
                detail=detail,
            )
        )

    return comparison
