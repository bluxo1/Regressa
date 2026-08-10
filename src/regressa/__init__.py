"""Regressa — prompt evaluation and regression-testing suite."""

__version__ = "0.1.0"

from regressa.config import Case, Suite, Target
from regressa.results import AssertionOutcome, CaseResult, RunSummary

__all__ = [
    "__version__",
    "Suite",
    "Case",
    "Target",
    "CaseResult",
    "AssertionOutcome",
    "RunSummary",
]
