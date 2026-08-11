"""Jitter measurement and tolerance recommendation."""

from __future__ import annotations

import pytest

from regressa.calibration import MIN_TOLERANCE, calibrate
from regressa.results import CaseResult, RunSummary, Usage


def result(case_id: str, score: float, passed: bool | None = None,
           target_id: str = "t", error: str | None = None) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        target_id=target_id,
        passed=(score >= 0.7) if passed is None else passed,
        score=score,
        error=error,
        usage=Usage(),
    )


def run(results: list[CaseResult], run_id: str = "r", suite_hash: str = "h") -> RunSummary:
    return RunSummary(
        run_id=run_id, suite="s", suite_hash=suite_hash, results=results
    ).recompute()


def test_needs_at_least_two_runs():
    cal = calibrate([run([result("a", 1.0)], run_id="r1")])
    assert cal.enough_data is False


def test_no_runs_at_all_is_handled():
    cal = calibrate([])
    assert cal.enough_data is False
    assert cal.cases == []


def test_identical_runs_show_no_jitter():
    runs = [run([result("a", 0.9)], run_id=f"r{i}") for i in range(3)]
    cal = calibrate(runs)
    assert cal.max_spread == pytest.approx(0.0)
    assert cal.jittery == []
    assert cal.flapping == []
    assert all(c.stable for c in cal.cases)


def test_spread_is_max_minus_min():
    runs = [
        run([result("a", 0.90)], run_id="r1"),
        run([result("a", 0.82)], run_id="r2"),
        run([result("a", 0.86)], run_id="r3"),
    ]
    cal = calibrate(runs)
    assert cal.cases[0].spread == pytest.approx(0.08)
    assert cal.cases[0].mean == pytest.approx(0.86)
    assert cal.cases[0].runs == 3


def test_a_case_that_flips_pass_fail_is_flapping_not_merely_jittery():
    runs = [
        run([result("a", 0.72, passed=True)], run_id="r1"),
        run([result("a", 0.68, passed=False)], run_id="r2"),
    ]
    cal = calibrate(runs)
    assert [c.case_id for c in cal.flapping] == ["a"]
    assert cal.jittery == []  # flapping cases are reported separately


def test_flapping_cases_do_not_inflate_the_recommended_tolerance():
    """A flapping case needs fixing; widening the gate for it would hide real
    regressions in every other case."""
    runs = [
        run([result("flappy", 1.0, passed=True), result("steady", 0.90)], run_id="r1"),
        run([result("flappy", 0.0, passed=False), result("steady", 0.88)], run_id="r2"),
    ]
    cal = calibrate(runs)
    # Driven by `steady`'s 0.02 spread, not `flappy`'s 1.0.
    assert cal.recommended_tolerance == pytest.approx(0.03)


def test_recommendation_adds_headroom_over_the_worst_observed_jitter():
    runs = [
        run([result("a", 0.90)], run_id="r1"),
        run([result("a", 0.80)], run_id="r2"),
    ]
    # 0.10 spread x 1.5 margin = 0.15
    assert calibrate(runs).recommended_tolerance == pytest.approx(0.15)


def test_recommendation_is_driven_by_the_worst_case_not_the_average():
    runs = [
        run([result("calm", 0.90), result("wild", 0.90)], run_id="r1"),
        run([result("calm", 0.90), result("wild", 0.70)], run_id="r2"),
    ]
    # The gate fires on the worst case, so sizing to the mean guarantees noise.
    assert calibrate(runs).recommended_tolerance == pytest.approx(0.30)


def test_recommendation_never_drops_below_the_floor():
    runs = [run([result("a", 0.9)], run_id=f"r{i}") for i in range(2)]
    assert calibrate(runs).recommended_tolerance == pytest.approx(MIN_TOLERANCE)


def test_recommendation_is_capped_at_one():
    runs = [
        run([result("a", 1.0, passed=True)], run_id="r1"),
        run([result("a", 0.0, passed=True)], run_id="r2"),
    ]
    assert calibrate(runs).recommended_tolerance <= 1.0


def test_runs_from_a_different_suite_definition_are_excluded():
    """An edited assertion is not jitter."""
    runs = [
        run([result("a", 0.90)], run_id="r1", suite_hash="new"),
        run([result("a", 0.90)], run_id="r2", suite_hash="new"),
        run([result("a", 0.10)], run_id="r3", suite_hash="old"),
    ]
    cal = calibrate(runs)
    assert cal.runs_compared == 2
    assert cal.runs_ignored == 1
    assert cal.max_spread == pytest.approx(0.0)


def test_errored_results_are_excluded_from_jitter():
    """A harness error has no meaningful score; counting it looks like a huge drop."""
    runs = [
        run([result("a", 0.90)], run_id="r1"),
        run([result("a", 0.0, error="connection reset")], run_id="r2"),
        run([result("a", 0.88)], run_id="r3"),
    ]
    cal = calibrate(runs)
    assert cal.cases[0].runs == 2
    assert cal.cases[0].spread == pytest.approx(0.02)


def test_cases_seen_in_only_one_run_are_ignored():
    runs = [
        run([result("both", 0.9), result("only-here", 0.5)], run_id="r1"),
        run([result("both", 0.9)], run_id="r2"),
    ]
    cal = calibrate(runs)
    assert [c.case_id for c in cal.cases] == ["both"]


def test_targets_are_tracked_separately():
    runs = [
        run([result("a", 0.9, target_id="v1"), result("a", 0.5, target_id="v2")], run_id="r1"),
        run([result("a", 0.9, target_id="v1"), result("a", 0.5, target_id="v2")], run_id="r2"),
    ]
    cal = calibrate(runs)
    assert {(c.case_id, c.target_id) for c in cal.cases} == {("a", "v1"), ("a", "v2")}
    assert cal.max_spread == pytest.approx(0.0)


def test_stdev_is_zero_for_a_single_pair_of_identical_scores():
    runs = [run([result("a", 0.5)], run_id=f"r{i}") for i in range(2)]
    assert calibrate(runs).cases[0].stdev == pytest.approx(0.0)
