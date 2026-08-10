"""Rich terminal output."""

from __future__ import annotations

import os
import sys
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from regressa.diffing import Change, RunDiff
from regressa.results import CaseResult, RunSummary


def _width() -> int | None:
    """Terminal width to render at.

    Rich falls back to 80 columns when stdout is not a TTY, which squeezes the
    results table until case ids and assertion messages are truncated — exactly
    the information you need when reading a CI log. Use a generous fixed width
    when piped, and let Rich detect the real size when interactive.
    """
    override = os.environ.get("REGRESSA_WIDTH", "")
    if override.isdigit():
        return int(override)
    return None if sys.stdout.isatty() else 160


_FANCY = {
    "ok": "✓", "no": "✗", "dash": "—", "up": "▲", "down": "▼",
    "dot": "·", "bar": "█", "empty": "░", "bullet": "•", "arrow": "→",
}
_PLAIN = {
    "ok": "+", "no": "x", "dash": "-", "up": "^", "down": "v",
    "dot": ".", "bar": "#", "empty": ".", "bullet": "*", "arrow": "->",
}


def _unicode_ok() -> bool:
    """Whether stdout can actually encode the glyphs we'd like to use.

    A Windows console on a legacy code page (cp1252) raises UnicodeEncodeError on
    a check mark, taking the whole command down with it. Detect that up front and
    degrade to ASCII rather than crashing on the last line of a passing run.
    """
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "".join(_FANCY.values()).encode(encoding)
    except (LookupError, UnicodeEncodeError, TypeError):
        return False
    return True


UNICODE = _unicode_ok()
_GLYPHS = _FANCY if UNICODE else _PLAIN


def glyph(name: str) -> str:
    return _GLYPHS[name]


# Rich normally picks a box style itself, but be explicit so ASCII mode is
# consistent with the glyph fallback above.
TABLE_BOX = box.HEAVY_HEAD if UNICODE else box.ASCII

console = Console(width=_width())

_CHANGE_STYLE = {
    Change.REGRESSED: "bold red",
    Change.SCORE_DROP: "yellow",
    Change.FIXED: "bold green",
    Change.SCORE_GAIN: "green",
    Change.ADDED: "cyan",
    Change.REMOVED: "magenta",
    Change.ERRORED: "bold yellow",
    Change.UNCHANGED: "dim",
}


def _status(result: CaseResult) -> Text:
    if result.errored:
        return Text("ERROR", style="bold yellow")
    if result.passed:
        return Text("PASS", style="green")
    return Text("FAIL", style="bold red")


def print_run(run: RunSummary, verbose: bool = False, show_passing: bool = False) -> None:
    table = Table(title=f"{run.suite} {glyph('dash')} run {run.run_id}",
                  title_style="bold", box=TABLE_BOX)
    table.add_column("", width=6)
    table.add_column("case")
    table.add_column("target")
    table.add_column("score", justify="right", width=6)
    table.add_column("ms", justify="right", width=7)
    table.add_column("cost", justify="right", width=10)
    table.add_column("detail", overflow="fold")

    for result in run.results:
        if result.passed and not show_passing and not verbose:
            continue
        table.add_row(
            _status(result),
            result.case_id,
            result.target_id,
            f"{result.score:.2f}",
            "cached" if result.cached else f"{result.latency_ms:.0f}",
            f"${result.usage.cost_usd:.5f}",
            Text(_detail(result)),
        )

    if table.row_count:
        console.print(table)
    elif run.total:
        console.print("[green]All cases passed.[/green] Use --show-passing to list them.")

    if verbose:
        for result in run.results:
            if not result.passed and result.output:
                console.print(
                    Panel(
                        Text(result.output[:2000]),
                        title=f"{result.case_id} / {result.target_id} output",
                        border_style="red",
                    )
                )

    print_summary(run)


def _detail(result: CaseResult) -> str:
    if result.errored:
        return result.error or ""
    failed = [a for a in result.assertions if not a.passed]
    if not failed:
        return f"{len(result.assertions)} assertion(s) passed"
    return "; ".join(f"{a.label or a.type}: {a.message}" for a in failed[:3])


def print_summary(run: RunSummary) -> None:
    parts = [
        f"[bold]{run.passed}/{run.total}[/bold] passed ({run.pass_rate:.0%})",
        f"score [bold]{run.score:.3f}[/bold]",
        f"cost [bold]${run.cost_usd:.4f}[/bold]",
    ]
    if run.errors:
        parts.append(f"[yellow]{run.errors} harness error(s)[/yellow]")
    if run.git_sha:
        parts.append(f"git {run.git_sha}")
    style = "green" if run.failed == 0 else "red"
    console.print(Panel(f"  {glyph('bullet')}  ".join(parts), border_style=style, title="summary"))


def print_diff(diff: RunDiff, show_unchanged: bool = False) -> None:
    header = (
        f"{diff.before.run_id} {glyph('arrow')} {diff.after.run_id}   "
        f"pass rate {diff.before.pass_rate:.0%} {glyph('arrow')} {diff.after.pass_rate:.0%} "
        f"({diff.pass_rate_delta:+.0%})   "
        f"score {diff.before.score:.3f} {glyph('arrow')} {diff.after.score:.3f} "
        f"({diff.score_delta:+.3f})   "
        f"cost {diff.cost_delta:+.4f} USD"
    )
    console.print(Panel(header, title="diff", border_style="cyan"))

    if diff.suite_changed:
        console.print(
            "[yellow]Warning:[/yellow] these runs came from different suite "
            "definitions, so some differences may be edits to the suite rather "
            "than model or prompt regressions."
        )

    table = Table(show_lines=False, box=TABLE_BOX)
    table.add_column("change", width=11)
    table.add_column("case")
    table.add_column("target")
    table.add_column("score", justify="right", width=18)
    table.add_column("detail", overflow="fold")

    ordering = [
        Change.REGRESSED,
        Change.SCORE_DROP,
        Change.ERRORED,
        Change.FIXED,
        Change.SCORE_GAIN,
        Change.ADDED,
        Change.REMOVED,
    ]
    if show_unchanged:
        ordering.append(Change.UNCHANGED)

    rows = 0
    for change in ordering:
        for delta in diff.by_change(change):
            before = glyph("dash") if delta.before_score is None else f"{delta.before_score:.2f}"
            after = glyph("dash") if delta.after_score is None else f"{delta.after_score:.2f}"
            table.add_row(
                Text(change.value, style=_CHANGE_STYLE[change]),
                delta.case_id,
                delta.target_id,
                f"{before} {glyph('arrow')} {after}",
                Text(delta.detail),
            )
            rows += 1

    if rows:
        console.print(table)
    else:
        console.print("[green]No changes between these runs.[/green]")

    if diff.has_regressions:
        console.print(
            f"[bold red]{len(diff.regressions)} regression(s).[/bold red] "
            "Non-zero exit for CI."
        )
    else:
        console.print("[green]No regressions.[/green]")


def print_runs(runs: list[RunSummary]) -> None:
    table = Table(title="run history", box=TABLE_BOX)
    # min_width + fold: a clipped run id can't be pasted into `regressa diff`.
    table.add_column("run id", min_width=22, overflow="fold")
    table.add_column("suite")
    table.add_column("when")
    table.add_column("pass", justify="right")
    table.add_column("score", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("git")
    table.add_column("label", overflow="fold")

    for run in runs:
        table.add_row(
            run.run_id,
            run.suite,
            run.started_at.strftime("%Y-%m-%d %H:%M"),
            f"{run.passed}/{run.total}",
            f"{run.score:.3f}",
            f"${run.cost_usd:.4f}",
            run.git_sha or glyph("dash"),
            run.label or glyph("dash"),
        )
    console.print(table)


def print_trend(suite: str, points: list[dict[str, Any]]) -> None:
    if not points:
        console.print(f"No runs recorded for suite {suite!r}.")
        return

    table = Table(
        title=f"{suite} {glyph('dash')} trend ({len(points)} runs, oldest first)",
        box=TABLE_BOX,
    )
    table.add_column("when")
    table.add_column("run id", min_width=22, overflow="fold")
    table.add_column("pass", justify="right")
    table.add_column("score", justify="right")
    table.add_column("delta", justify="right", width=8)
    table.add_column("", width=20)
    table.add_column("cost", justify="right")
    table.add_column("label", overflow="fold")

    previous: float | None = None
    for point in points:
        score = point["score"]
        table.add_row(
            point["started_at"][:16].replace("T", " "),
            point["run_id"],
            f"{point['passed']}/{point['total']}",
            f"{score:.3f}",
            _delta(score, previous),
            _sparkbar(score),
            f"${point['cost_usd']:.4f}",
            point.get("label") or glyph("dash"),
        )
        previous = score
    console.print(table)


def _delta(score: float, previous: float | None) -> Text:
    if previous is None:
        return Text(glyph("dot"), style="dim")
    change = score - previous
    if change > 1e-9:
        return Text(f"{glyph('up')}{change:+.3f}", style="green")
    if change < -1e-9:
        return Text(f"{glyph('down')}{change:+.3f}", style="red")
    return Text(glyph("dot"), style="dim")


def _sparkbar(value: float, width: int = 20) -> Text:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    style = "green" if value >= 0.9 else "yellow" if value >= 0.7 else "red"
    return Text(glyph("bar") * filled + glyph("empty") * (width - filled), style=style)
