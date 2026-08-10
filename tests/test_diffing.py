from __future__ import annotations

import pytest

from regressa.diffing import Change, diff_runs
from regressa.results import CaseResult, RunSummary, Usage


def result(case_id="c", target_id="t", passed=True, score=1.0, error=None, assertions=None):
    return CaseResult(
        case_id=case_id,
        target_id=target_id,
        passed=passed,
        score=score,
        error=error,
        assertions=assertions or [],
        usage=Usage(),
    )


def run(results, run_id="r", suite="s", suite_hash="h"):
    summary = RunSummary(run_id=run_id, suite=suite, suite_hash=suite_hash, results=results)
    return summary.recompute()


def test_pass_to_fail_is_a_regression():
    diff = diff_runs(run([result(passed=True)]), run([result(passed=False, score=0.0)]))
    assert diff.has_regressions
    assert diff.deltas[0].change is Change.REGRESSED


def test_fail_to_pass_is_a_fix_not_a_regression():
    diff = diff_runs(run([result(passed=False, score=0.0)]), run([result(passed=True)]))
    assert not diff.has_regressions
    assert diff.deltas[0].change is Change.FIXED


def test_small_score_drop_within_tolerance_is_noise():
    diff = diff_runs(
        run([result(score=0.90)]), run([result(score=0.88)]), tolerance=0.05
    )
    assert diff.deltas[0].change is Change.UNCHANGED
    assert not diff.has_regressions


def test_large_score_drop_is_a_regression_even_while_still_passing():
    diff = diff_runs(
        run([result(score=0.95)]), run([result(score=0.60)]), tolerance=0.05
    )
    assert diff.deltas[0].change is Change.SCORE_DROP
    assert diff.has_regressions


def test_harness_error_is_never_counted_as_a_regression():
    diff = diff_runs(
        run([result(passed=True)]),
        run([result(passed=False, score=0.0, error="connection reset")]),
    )
    assert diff.deltas[0].change is Change.ERRORED
    assert not diff.has_regressions
    assert diff.errors


def test_a_new_failing_case_is_added_not_regressed():
    diff = diff_runs(run([]), run([result(passed=False, score=0.0)]))
    assert diff.deltas[0].change is Change.ADDED
    assert not diff.has_regressions


def test_a_removed_case_is_reported():
    diff = diff_runs(run([result()]), run([]))
    assert diff.deltas[0].change is Change.REMOVED


def test_suite_hash_change_is_surfaced_as_a_caveat():
    diff = diff_runs(
        run([result()], suite_hash="aaa"), run([result()], suite_hash="bbb")
    )
    assert diff.suite_changed is True


def test_regression_detail_names_the_failing_assertion():
    from regressa.results import AssertionOutcome

    failing = AssertionOutcome(
        type="contains", label="states the window", passed=False, score=0.0,
        message="missing ['30 day']",
    )
    diff = diff_runs(
        run([result(passed=True)]),
        run([result(passed=False, score=0.0, assertions=[failing])]),
    )
    assert "states the window" in diff.deltas[0].detail


def test_aggregate_deltas():
    before = run([result(score=0.5, passed=False), result(target_id="u", score=1.0)])
    after = run([result(score=1.0), result(target_id="u", score=1.0)])
    diff = diff_runs(before, after)
    assert diff.pass_rate_delta == pytest.approx(0.5)
    assert diff.score_delta == pytest.approx(0.25)
