"""Regressa command-line interface.

Exit codes are the contract CI depends on:
    0  everything passed (and no regressions vs baseline, when checked)
    1  assertion failures, or regressions vs baseline
    2  configuration / usage error
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from regressa import __version__, capabilities, loader, reporting
from regressa.diffing import DEFAULT_TOLERANCE, diff_runs
from regressa.loader import SuiteLoadError
from regressa.reporting.console import _width, console, glyph
from regressa.results import RunSummary
from regressa.runner import ConfigError, Runner, RunOptions
from regressa.storage import Store

app = typer.Typer(
    name="regressa",
    help="Prompt evaluation and regression-testing suite.",
    no_args_is_help=True,
    add_completion=False,
)
baseline_app = typer.Typer(help="Manage the comparison baseline for a suite.")
app.add_typer(baseline_app, name="baseline")

err = Console(stderr=True, width=_width())

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2


def _fail(message: str) -> None:
    err.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(EXIT_CONFIG)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    # invoke_without_command is required for `--version` to be reachable: with
    # the default, Click rejects a bare `regressa --version` as a missing command
    # before the callback ever runs.
    if version:
        console.print(f"regressa {__version__}")
        raise typer.Exit(EXIT_OK)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(EXIT_OK)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


@app.command()
def run(
    paths: Optional[list[str]] = typer.Argument(
        None, help="Suite files or directories. Defaults to ./suites, then ./"
    ),
    db: Optional[str] = typer.Option(None, "--db", help="SQLite history file."),
    target: Optional[list[str]] = typer.Option(
        None, "--target", "-t", help="Only run these target ids."
    ),
    tag: Optional[list[str]] = typer.Option(
        None, "--tag", help="Only run cases carrying one of these tags."
    ),
    case: Optional[list[str]] = typer.Option(
        None, "--case", "-k", help="Only run these case ids."
    ),
    var: Optional[list[str]] = typer.Option(
        None, "--var", help="Override a suite variable: --var key=value (repeatable)."
    ),
    concurrency: int = typer.Option(4, "--concurrency", "-j", min=1, max=32),
    cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Reuse identical prior responses."
    ),
    label: Optional[str] = typer.Option(None, "--label", help="Name this run."),
    check_baseline: bool = typer.Option(
        True,
        "--check-baseline/--no-check-baseline",
        help="Compare against the stored baseline and fail on regressions.",
    ),
    set_baseline: bool = typer.Option(
        False, "--set-baseline", help="Make this run the new baseline."
    ),
    tolerance: float = typer.Option(
        DEFAULT_TOLERANCE, "--tolerance", help="Score drop treated as noise."
    ),
    fail_under: Optional[float] = typer.Option(
        None, "--fail-under", help="Fail if the mean score is below this (0-1)."
    ),
    html_out: Optional[str] = typer.Option(None, "--html", help="Write an HTML report."),
    json_out: Optional[str] = typer.Option(None, "--json", help="Write results as JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show failing outputs."),
    show_passing: bool = typer.Option(False, "--show-passing"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate suites and print the plan without calling models."
    ),
) -> None:
    """Run one or more suites."""
    files = loader.discover(paths)
    if not files:
        _fail("no suite files found. Pass a path, or create ./suites/*.yaml")

    overrides: dict[str, str] = {}
    for item in var or []:
        if "=" not in item:
            _fail(f"--var expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value

    store = Store(db)
    runner = Runner(store)
    exit_code = EXIT_OK

    try:
        for path in files:
            try:
                suite = loader.load_suite(path)
            except SuiteLoadError as exc:
                _fail(str(exc))

            issues = suite.validate_targets()
            if issues:
                for issue in issues:
                    err.print(f"[red]{glyph('no')}[/red] ({issue.target_id}) {issue.message}")
                raise typer.Exit(EXIT_CONFIG)

            if dry_run:
                _print_plan(suite, path)
                continue

            console.print(f"[bold]{suite.name}[/bold]  ({path})")
            try:
                summary = runner.run(
                    suite,
                    RunOptions(
                        concurrency=concurrency,
                        use_cache=cache,
                        targets=target,
                        tags=tag,
                        cases=case,
                        label=label,
                        var_overrides=overrides,
                    ),
                )
            except ConfigError as exc:
                _fail(str(exc))

            baseline = store.get_baseline(suite.name) if check_baseline else None
            diff = (
                diff_runs(baseline, summary, tolerance=tolerance)
                if baseline is not None and baseline.run_id != summary.run_id
                else None
            )

            reporting.print_run(summary, verbose=verbose, show_passing=show_passing)
            if diff is not None:
                reporting.print_diff(diff)
            elif check_baseline and not set_baseline:
                console.print(
                    "[dim]No baseline stored for this suite - "
                    "run with --set-baseline to establish one.[/dim]"
                )

            if html_out:
                written = reporting.write_report(
                    html_out, summary, trend=store.trend(suite.name), diff=diff
                )
                console.print(f"HTML report written to {written}")
            if json_out:
                console.print(f"JSON results written to {reporting.write_json(json_out, summary)}")

            if set_baseline:
                store.set_baseline(suite.name, summary.run_id)
                console.print(f"Baseline for {suite.name!r} set to {summary.run_id}")

            if summary.failed:
                exit_code = EXIT_FAILED
            if diff is not None and diff.has_regressions:
                exit_code = EXIT_FAILED
            if fail_under is not None and summary.score < fail_under:
                err.print(
                    f"[red]score {summary.score:.3f} is below --fail-under "
                    f"{fail_under:.3f}[/red]"
                )
                exit_code = EXIT_FAILED
    finally:
        store.close()

    raise typer.Exit(exit_code)


def _print_plan(suite, path: Path) -> None:  # type: ignore[no-untyped-def]
    console.print(f"[bold]{suite.name}[/bold]  ({path})")
    total = 0
    for case in suite.cases:
        targets = [t.id for t in suite.resolved_targets(case)]
        n_assertions = len(suite.resolved_assertions(case))
        total += len(targets)
        console.print(
            f"  {case.id}  -> targets={','.join(targets)}  "
            f"assertions={n_assertions}  tags={','.join(case.tags) or glyph('dash')}"
        )
    console.print(f"  [dim]{total} case/target pair(s); no model calls made.[/dim]")


# --------------------------------------------------------------------------- #
# diff / history / trend / show
# --------------------------------------------------------------------------- #


@app.command()
def diff(
    before: str = typer.Argument(..., help="Run id, `baseline`, `latest`, or `latest~N`."),
    after: str = typer.Argument("latest", help="Run id, `latest`, or `latest~N`."),
    suite: Optional[str] = typer.Option(
        None, "--suite", "-s", help="Needed to resolve `baseline` / `latest`."
    ),
    db: Optional[str] = typer.Option(None, "--db"),
    tolerance: float = typer.Option(DEFAULT_TOLERANCE, "--tolerance"),
    show_unchanged: bool = typer.Option(False, "--show-unchanged"),
) -> None:
    """Compare two runs and report regressions. Exits 1 if any are found."""
    with Store(db) as store:
        before_run = _resolve(store, before, suite)
        after_run = _resolve(store, after, suite)
        result = diff_runs(before_run, after_run, tolerance=tolerance)
        reporting.print_diff(result, show_unchanged=show_unchanged)
    raise typer.Exit(EXIT_FAILED if result.has_regressions else EXIT_OK)


def _resolve(store: Store, ref: str, suite: str | None) -> RunSummary:
    resolved = store.resolve_run(ref, suite)
    if resolved is None:
        hint = ""
        if ref in ("baseline", "latest") and suite is None:
            hint = " - pass --suite so it can be resolved"
        _fail(f"could not resolve run {ref!r}{hint}")
    return resolved  # type: ignore[return-value]


@app.command(name="history")
def history_cmd(
    suite: Optional[str] = typer.Option(None, "--suite", "-s"),
    limit: int = typer.Option(20, "--limit", "-n", min=1),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """List recorded runs, newest first."""
    with Store(db) as store:
        runs = store.list_runs(suite=suite, limit=limit)
        if not runs:
            console.print("No runs recorded yet.")
            raise typer.Exit(EXIT_OK)
        reporting.print_runs(runs)


@app.command()
def trend(
    suite: Optional[str] = typer.Option(None, "--suite", "-s"),
    limit: int = typer.Option(30, "--limit", "-n", min=2),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Show score movement across runs of a suite."""
    with Store(db) as store:
        names = [suite] if suite else store.suites()
        if not names:
            console.print("No runs recorded yet.")
            raise typer.Exit(EXIT_OK)
        for name in names:
            reporting.print_trend(name, store.trend(name, limit=limit))


@app.command()
def show(
    run_id: str = typer.Argument("latest"),
    suite: Optional[str] = typer.Option(None, "--suite", "-s"),
    db: Optional[str] = typer.Option(None, "--db"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    show_passing: bool = typer.Option(True, "--show-passing/--failures-only"),
) -> None:
    """Print a stored run."""
    with Store(db) as store:
        run = _resolve(store, run_id, suite)
        reporting.print_run(run, verbose=verbose, show_passing=show_passing)


@app.command()
def report(
    run_id: str = typer.Argument("latest"),
    out: str = typer.Option("regressa-report.html", "--out", "-o"),
    suite: Optional[str] = typer.Option(None, "--suite", "-s"),
    db: Optional[str] = typer.Option(None, "--db"),
    compare_baseline: bool = typer.Option(True, "--compare-baseline/--no-compare-baseline"),
) -> None:
    """Write an HTML report for a stored run."""
    with Store(db) as store:
        run = _resolve(store, run_id, suite)
        result_diff = None
        if compare_baseline:
            baseline = store.get_baseline(run.suite)
            if baseline is not None and baseline.run_id != run.run_id:
                result_diff = diff_runs(baseline, run)
        written = reporting.write_report(
            out, run, trend=store.trend(run.suite), diff=result_diff
        )
    console.print(f"HTML report written to {written}")


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #


@baseline_app.command("set")
def baseline_set(
    run_id: str = typer.Argument("latest"),
    suite: Optional[str] = typer.Option(None, "--suite", "-s"),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Pin a run as the baseline future runs are compared against."""
    with Store(db) as store:
        run = _resolve(store, run_id, suite)
        store.set_baseline(run.suite, run.run_id)
    console.print(f"Baseline for {run.suite!r} set to {run.run_id}")


@baseline_app.command("show")
def baseline_show(
    suite: Optional[str] = typer.Option(None, "--suite", "-s"),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Show the current baseline for each suite."""
    with Store(db) as store:
        names = [suite] if suite else store.suites()
        if not names:
            console.print("No runs recorded yet.")
            raise typer.Exit(EXIT_OK)
        for name in names:
            baseline = store.get_baseline(name)
            if baseline is None:
                console.print(f"{name}: [dim]no baseline set[/dim]")
            else:
                console.print(
                    f"{name}: {baseline.run_id} "
                    f"({baseline.passed}/{baseline.total} passed, "
                    f"score {baseline.score:.3f})"
                )


@baseline_app.command("clear")
def baseline_clear(
    suite: str = typer.Argument(...),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Remove a suite's baseline."""
    with Store(db) as store:
        removed = store.clear_baseline(suite)
    console.print(
        f"Baseline cleared for {suite!r}" if removed else f"No baseline was set for {suite!r}"
    )


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #


@app.command("cache-clear")
def cache_clear(db: Optional[str] = typer.Option(None, "--db")) -> None:
    """Drop every cached model response."""
    with Store(db) as store:
        count = store.cache_clear()
    console.print(f"Cleared {count} cached response(s).")


@app.command()
def validate(
    paths: Optional[list[str]] = typer.Argument(None),
) -> None:
    """Check suites for schema errors and model-capability mismatches."""
    files = loader.discover(paths)
    if not files:
        _fail("no suite files found")

    problems = 0
    for path in files:
        try:
            suite = loader.load_suite(path)
        except SuiteLoadError as exc:
            err.print(f"[red]{glyph('no')}[/red] {exc}")
            problems += 1
            continue

        issues = suite.validate_targets()
        for issue in issues:
            err.print(f"[red]{glyph('no')}[/red] {path} ({issue.target_id}) {issue.message}")
        problems += len(issues)
        if not issues:
            pairs = sum(len(suite.resolved_targets(c)) for c in suite.cases)
            console.print(
                f"[green]{glyph('ok')}[/green] {path} {suite.name}: "
                f"{len(suite.cases)} case(s), {len(suite.targets)} target(s), "
                f"{pairs} pair(s)"
            )

    raise typer.Exit(EXIT_CONFIG if problems else EXIT_OK)


@app.command()
def models() -> None:
    """List models Regressa knows how to price and validate."""
    from rich.table import Table

    table = Table(title="known models")
    table.add_column("model")
    table.add_column("in $/MTok", justify="right")
    table.add_column("out $/MTok", justify="right")
    table.add_column("context", justify="right")
    table.add_column("max out", justify="right")
    table.add_column("sampling")
    table.add_column("effort")
    table.add_column("notes", overflow="fold")

    for name in capabilities.known_models():
        caps = capabilities.get(name)
        table.add_row(
            name,
            f"{caps.input_per_mtok:.2f}",
            f"{caps.output_per_mtok:.2f}",
            f"{caps.context_window:,}",
            f"{caps.max_output_tokens:,}",
            "[green]yes[/green]" if caps.supports_sampling else "[red]rejected[/red]",
            ", ".join(caps.effort_levels) or glyph("dash"),
            caps.notes or "",
        )
    console.print(table)
    console.print(
        "[dim]`sampling: rejected` means temperature/top_p return a 400 on that "
        "model - Regressa refuses to send them and flags the target instead.[/dim]"
    )
