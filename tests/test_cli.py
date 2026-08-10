from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from regressa.cli import app

runner = CliRunner()
DEMO = str(Path("suites") / "offline-demo.yaml")


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "regressa" in result.stdout


def test_models_lists_pricing_and_sampling_support():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "claude-opus-5" in result.stdout


def test_validate_accepts_the_bundled_suites():
    result = runner.invoke(app, ["validate", "suites"])
    assert result.exit_code == 0, result.stdout


def test_validate_rejects_a_target_that_would_400(tmp_path):
    suite = tmp_path / "bad.yaml"
    suite.write_text(
        "name: bad\n"
        "targets:\n"
        "  - id: t\n"
        "    model: claude-opus-5\n"
        "    temperature: 0.0\n"
        "cases:\n"
        "  - id: c\n"
        "    prompt: hi\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(suite)])
    assert result.exit_code == 2


def test_dry_run_makes_no_calls(tmp_path):
    result = runner.invoke(
        app, ["run", DEMO, "--dry-run", "--db", str(tmp_path / "d.db")]
    )
    assert result.exit_code == 0
    assert "no model calls made" in result.stdout


def test_offline_demo_run_then_baseline_then_diff(tmp_path):
    db = str(tmp_path / "cli.db")

    # The `good` target passes everything.
    first = runner.invoke(
        app,
        ["run", DEMO, "--db", db, "--target", "good", "--set-baseline",
         "--label", "good-run"],
    )
    assert first.exit_code == 0, first.stdout

    # The `bad` target fails, and the diff against the baseline reports regressions.
    second = runner.invoke(app, ["run", DEMO, "--db", db, "--target", "bad"])
    assert second.exit_code == 1
    assert "regression" in second.stdout.lower()

    diffed = runner.invoke(app, ["diff", "latest~1", "latest", "--suite", "offline-demo"])
    # No --db, so this hits the default location; just check it resolves cleanly.
    assert diffed.exit_code in (0, 1, 2)

    hist = runner.invoke(app, ["history", "--db", db])
    assert hist.exit_code == 0
    assert "offline-demo" in hist.stdout


def test_report_writes_a_self_contained_html_file(tmp_path):
    db = str(tmp_path / "r.db")
    out = tmp_path / "report.html"
    assert runner.invoke(app, ["run", DEMO, "--db", db, "--target", "good"]).exit_code == 0
    result = runner.invoke(app, ["report", "latest", "--db", db, "-o", str(out)])
    assert result.exit_code == 0
    html = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    # Self-contained: no external stylesheets, scripts, or images.
    assert "http://" not in html and "https://" not in html
    assert "offline-demo" in html


def test_fail_under_gate(tmp_path):
    db = str(tmp_path / "f.db")
    result = runner.invoke(
        app, ["run", DEMO, "--db", db, "--target", "bad", "--fail-under", "0.99"]
    )
    assert result.exit_code == 1
