"""Report renderers."""

from regressa.reporting.console import (
    print_comparison,
    print_diff,
    print_run,
    print_runs,
    print_summary,
    print_trend,
)
from regressa.reporting.html import render_report, write_json, write_report

__all__ = [
    "print_run",
    "print_comparison",
    "print_diff",
    "print_runs",
    "print_summary",
    "print_trend",
    "render_report",
    "write_report",
    "write_json",
]
