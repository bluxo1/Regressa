"""Measure run-to-run jitter and recommend a `--tolerance`.

`diff` treats a score drop larger than `--tolerance` as a regression. The default
of 0.05 is a guess: judge assertions vary slightly between identical runs, and how
much depends on your rubrics, your grader model, and your prompts. Guessing too
tight produces phantom regressions; too loose hides real ones.

This measures it from runs you already have, so it needs no extra API calls.

Two distinct problems are reported, and the second matters more:

* **Score jitter** — the same case scoring 0.82 then 0.79. A tolerance absorbs it.
* **Flapping** — the same case passing in one run and failing in the next. No
  tolerance fixes that, because pass/fail is not a threshold on the score. A
  flapping case will keep manufacturing regressions until its assertions are
  tightened or its rubric is made less ambiguous.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from regressa.results import RunSummary

# Only compare runs built from the same suite definition; otherwise the "jitter"
# includes your own edits.
MIN_RUNS = 2

# Never recommend a tolerance below this. Floating-point noise and one-off
# rounding shouldn't drive the gate to zero.
MIN_TOLERANCE = 0.01

# Headroom over the worst observed jitter, so a slightly unluckier run than the
# ones sampled doesn't immediately trip the gate.
SAFETY_MARGIN = 1.5


@dataclass
class CaseJitter:
    case_id: str
    target_id: str
    scores: list[float] = field(default_factory=list)
    pass_states: list[bool] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.scores)

    @property
    def spread(self) -> float:
        return max(self.scores) - min(self.scores) if self.scores else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def flapping(self) -> bool:
        """Passed in at least one run and failed in at least one other."""
        return len(set(self.pass_states)) > 1

    @property
    def stable(self) -> bool:
        return not self.flapping and self.spread == 0.0


@dataclass
class Calibration:
    suite: str
    suite_hash: str
    runs_compared: int
    runs_ignored: int
    cases: list[CaseJitter] = field(default_factory=list)

    @property
    def flapping(self) -> list[CaseJitter]:
        return [c for c in self.cases if c.flapping]

    @property
    def jittery(self) -> list[CaseJitter]:
        """Cases whose score moved without flipping pass/fail, worst first."""
        return sorted(
            (c for c in self.cases if not c.flapping and c.spread > 0),
            key=lambda c: c.spread,
            reverse=True,
        )

    @property
    def max_spread(self) -> float:
        return max((c.spread for c in self.cases), default=0.0)

    @property
    def recommended_tolerance(self) -> float:
        """Worst observed jitter plus headroom, rounded up to 2dp.

        Deliberately based on the maximum rather than a mean: the gate fires on
        the worst case, so sizing it to the average guarantees false positives.
        Flapping cases are excluded — they need fixing, not a wider tolerance,
        and letting them inflate the gate would mask real regressions elsewhere.
        """
        non_flapping = [c.spread for c in self.cases if not c.flapping]
        worst = max(non_flapping, default=0.0)
        raw = max(worst * SAFETY_MARGIN, MIN_TOLERANCE)
        # Round before the ceiling: 0.02 * 1.5 is 0.030000000000000006 in binary
        # floating point, and ceiling that raw would bump the recommendation a
        # whole step to 0.04 on nothing but representation noise.
        return min(1.0, math.ceil(round(raw, 9) * 100) / 100)

    @property
    def enough_data(self) -> bool:
        return self.runs_compared >= MIN_RUNS


def calibrate(runs: list[RunSummary]) -> Calibration:
    """Compare runs sharing the newest run's suite definition.

    `runs` should be newest-first, as `Store.list_runs` returns them. Runs from a
    different suite definition are counted in `runs_ignored` and excluded, because
    an edited assertion is not jitter.
    """
    if not runs:
        return Calibration(suite="", suite_hash="", runs_compared=0, runs_ignored=0)

    newest = runs[0]
    comparable = [r for r in runs if r.suite_hash == newest.suite_hash]

    result = Calibration(
        suite=newest.suite,
        suite_hash=newest.suite_hash,
        runs_compared=len(comparable),
        runs_ignored=len(runs) - len(comparable),
    )

    buckets: dict[tuple[str, str], CaseJitter] = {}
    for run in comparable:
        for r in run.results:
            # A harness error has no meaningful score; including it would look
            # like a huge drop and poison the recommendation.
            if r.errored:
                continue
            bucket = buckets.setdefault(
                r.key, CaseJitter(case_id=r.case_id, target_id=r.target_id)
            )
            bucket.scores.append(r.score)
            bucket.pass_states.append(r.passed)

    # Only pairs seen in more than one run say anything about variance.
    result.cases = sorted(
        (c for c in buckets.values() if c.runs >= MIN_RUNS),
        key=lambda c: (c.case_id, c.target_id),
    )
    return result
