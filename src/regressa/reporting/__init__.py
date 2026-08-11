"""Report renderers."""

from regressa.reporting.console import (
    print_calibration,
    print_comparison,
    print_diagnosis,
    print_diff,
    print_run,
    print_runs,
    print_summary,
    print_trend,
)
from regressa.reporting.html import (
    comparison_to_dict,
    render_report,
    write_comparison_json,
    write_json,
    write_report,
)

__all__ = [
    "print_run",
    "print_comparison",
    "print_calibration",
    "print_diagnosis",
    "print_diff",
    "print_runs",
    "print_summary",
    "print_trend",
    "render_report",
    "write_report",
    "write_json",
    "write_comparison_json",
    "comparison_to_dict",
]
