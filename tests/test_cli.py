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


# Overriding this var makes the `greeting` case fail while every other case and
# the target id stay put — which is what produces a real regression rather than
# an added/removed pair.
DEGRADE = ["--var", "greeting_reply=As an AI language model, I cannot assist."]


def test_baseline_then_regression_is_detected_and_exits_nonzero(tmp_path):
    db = str(tmp_path / "cli.db")

    first = runner.invoke(
        app, ["run", DEMO, "--db", db, "--set-baseline", "--label", "known-good"]
    )
    assert first.exit_code == 0, first.stdout

    second = runner.invoke(app, ["run", DEMO, "--db", db, *DEGRADE])
    assert second.exit_code == 1
    assert "regressed" in second.stdout
    assert "greeting" in second.stdout

    # An explicit diff of the same two runs reaches the same conclusion.
    diffed = runner.invoke(
        app, ["diff", "baseline", "latest", "--suite", "offline-demo", "--db", db]
    )
    assert diffed.exit_code == 1
    assert "regressed" in diffed.stdout

    hist = runner.invoke(app, ["history", "--db", db])
    assert hist.exit_code == 0
    assert "offline-demo" in hist.stdout


def test_var_override_is_recorded_in_the_run_label(tmp_path):
    # Otherwise a diff between two differently-parameterised runs would look like
    # an unexplained regression.
    db = str(tmp_path / "v.db")
    runner.invoke(app, ["run", DEMO, "--db", db, *DEGRADE])
    hist = runner.invoke(app, ["history", "--db", db])
    assert "greeting_reply" in hist.stdout


def test_bad_var_syntax_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["run", DEMO, "--db", str(tmp_path / "x.db"), "--var", "oops"])
    assert result.exit_code == 2


def test_report_writes_a_self_contained_html_file(tmp_path):
    db = str(tmp_path / "r.db")
    out = tmp_path / "report.html"
    assert runner.invoke(app, ["run", DEMO, "--db", db]).exit_code == 0
    result = runner.invoke(app, ["report", "latest", "--db", db, "-o", str(out)])
    assert result.exit_code == 0
    html = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    # Self-contained: no external stylesheets, scripts, fonts, or images.
    assert "http://" not in html and "https://" not in html
    assert "offline-demo" in html


def test_html_report_includes_a_trend_chart_once_there_are_two_runs(tmp_path):
    db = str(tmp_path / "t.db")
    out = tmp_path / "r.html"
    runner.invoke(app, ["run", DEMO, "--db", db])
    runner.invoke(app, ["run", DEMO, "--db", db, *DEGRADE])
    runner.invoke(app, ["report", "latest", "--db", db, "-o", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "<svg" in html and "polyline" in html


def test_fail_under_gate(tmp_path):
    db = str(tmp_path / "f.db")
    result = runner.invoke(app, ["run", DEMO, "--db", db, *DEGRADE, "--fail-under", "0.99"])
    assert result.exit_code == 1
