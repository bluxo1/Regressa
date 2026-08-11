"""A/B comparison of two targets within one run."""

from __future__ import annotations

import pytest

from regressa.diffing import Verdict, compare_targets
from regressa.results import CaseResult, RunSummary, Usage


def result(case_id: str, target_id: str, passed: bool, score: float | None = None,
           assertions=None) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        target_id=target_id,
        passed=passed,
        score=(1.0 if passed else 0.0) if score is None else score,
        assertions=assertions or [],
        usage=Usage(),
    )


def run(results: list[CaseResult]) -> RunSummary:
    return RunSummary(
        run_id="r1", suite="s", suite_hash="h", results=results
    ).recompute()


def test_candidate_failing_what_the_baseline_passes_is_flagged():
    summary = run([
        result("escalate", "v1", passed=True),
        result("escalate", "v2", passed=False),
    ])
    cmp = compare_targets(summary, "v1", "v2")
    assert cmp.deltas[0].verdict is Verdict.A_ONLY
    assert cmp.b_is_worse is True


def test_candidate_fixing_a_baseline_failure_is_an_improvement():
    summary = run([
        result("tone", "v1", passed=False),
        result("tone", "v2", passed=True),
    ])
    cmp = compare_targets(summary, "v1", "v2")
    assert cmp.deltas[0].verdict is Verdict.B_ONLY
    assert cmp.b_is_worse is False
    assert len(cmp.b_wins) == 1


def test_both_passing_is_a_tie():
    summary = run([
        result("c", "v1", passed=True),
        result("c", "v2", passed=True),
    ])
    assert compare_targets(summary, "v1", "v2").deltas[0].verdict is Verdict.BOTH_PASS


def test_both_failing_is_reported_but_is_not_a_candidate_regression():
    summary = run([
        result("c", "v1", passed=False),
        result("c", "v2", passed=False),
    ])
    cmp = compare_targets(summary, "v1", "v2")
    assert cmp.deltas[0].verdict is Verdict.BOTH_FAIL
    assert cmp.b_is_worse is False


def test_a_mixed_result_reports_wins_on_both_sides():
    summary = run([
        result("keeps", "v1", passed=True), result("keeps", "v2", passed=False),
        result("fixes", "v1", passed=False), result("fixes", "v2", passed=True),
        result("tie", "v1", passed=True), result("tie", "v2", passed=True),
    ])
    cmp = compare_targets(summary, "v1", "v2")
    assert [d.case_id for d in cmp.a_wins] == ["keeps"]
    assert [d.case_id for d in cmp.b_wins] == ["fixes"]
    # A candidate that fixes one thing and breaks another is still not safe.
    assert cmp.b_is_worse is True


def test_mean_scores_are_reported_per_target():
    summary = run([
        result("a", "v1", passed=True, score=1.0), result("a", "v2", passed=False, score=0.5),
        result("b", "v1", passed=True, score=0.8), result("b", "v2", passed=True, score=0.9),
    ])
    cmp = compare_targets(summary, "v1", "v2")
    assert cmp.a_score == pytest.approx(0.9)
    assert cmp.b_score == pytest.approx(0.7)


def test_a_case_run_against_only_one_target_is_marked_missing():
    summary = run([
        result("only-v1", "v1", passed=True),
        result("shared", "v1", passed=True),
        result("shared", "v2", passed=True),
    ])
    cmp = compare_targets(summary, "v1", "v2")
    missing = cmp.by_verdict(Verdict.B_MISSING)
    assert [d.case_id for d in missing] == ["only-v1"]
    # A case the candidate never ran is not evidence the candidate is worse.
    assert cmp.b_is_worse is False


def test_an_unknown_target_id_raises_rather_than_reporting_everything_missing():
    summary = run([result("c", "v1", passed=True)])
    with pytest.raises(ValueError) as exc:
        compare_targets(summary, "v1", "typo")
    assert "typo" in str(exc.value)
    assert "v1" in str(exc.value)


def test_detail_names_the_assertion_the_candidate_failed():
    from regressa.results import AssertionOutcome

    failing = AssertionOutcome(
        type="contains", label="exact escalation phrase", passed=False, score=0.0,
        message="missing ['escalating to a specialist']",
    )
    summary = run([
        result("escalate", "v1", passed=True),
        result("escalate", "v2", passed=False, assertions=[failing]),
    ])
    cmp = compare_targets(summary, "v1", "v2")
    assert "exact escalation phrase" in cmp.deltas[0].detail
