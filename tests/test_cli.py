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


def test_target_ids_survive_rich_markup_parsing(tmp_path):
    # A target id like `t` inside square brackets would be eaten as a Rich tag,
    # hiding which target is misconfigured.
    suite = tmp_path / "ids.yaml"
    suite.write_text(
        "name: ids\n"
        "targets:\n"
        "  - id: bold\n"
        "    model: claude-opus-5\n"
        "    temperature: 0.0\n"
        "cases:\n"
        "  - id: c\n"
        "    prompt: hi\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(suite)])
    assert result.exit_code == 2
    assert "bold" in result.output


def test_model_output_containing_markup_is_not_interpreted(tmp_path):
    # Model output is untrusted for markup: "[/]" would otherwise raise, and
    # "[red]" would be swallowed instead of shown.
    suite = tmp_path / "markup.yaml"
    suite.write_text(
        "name: markup\n"
        "targets:\n"
        "  - id: m\n"
        "    provider: mock\n"
        "    model: mock-1\n"
        "    options:\n"
        "      default: \"[red]not a tag[/] [bold]\"\n"
        "cases:\n"
        "  - id: c\n"
        "    prompt: hi\n"
        "    assertions:\n"
        "      - type: contains\n"
        "        value: absent-string\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["run", str(suite), "--db", str(tmp_path / "m.db"), "-v"]
    )
    assert result.exit_code == 1
    # A clean non-zero exit raises SystemExit; anything else (e.g. rich's
    # MarkupError on the unbalanced "[/]") means the output was parsed as markup.
    assert isinstance(result.exception, SystemExit)
    assert "not a tag" in result.output


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


# Two prompt versions as two mock targets: v2 fixes `tone` but breaks
# `escalate` — the mixed result a real prompt rewrite usually produces.
AB_SUITE = """
name: ab
targets:
  - id: v1-live
    provider: mock
    model: mock-1
    options:
      by_case:
        escalate: "escalating to a specialist now"
        tone: "We sincerely apologize for any inconvenience."
  - id: v2-candidate
    provider: mock
    model: mock-1
    options:
      by_case:
        escalate: "handing you to a specialist now"
        tone: "That is frustrating and I am fixing it now."
  # Fails everything, so comparisons against it are asymmetric and can prove
  # that direction is respected.
  - id: v0-broken
    provider: mock
    model: mock-1
    options:
      by_case:
        escalate: "have you tried turning it off and on again"
        tone: "We sincerely apologize for any inconvenience."
cases:
  - id: escalate
    prompt: "we were breached"
    assertions:
      - type: contains
        label: exact escalation phrase
        value: "escalating to a specialist"
  - id: tone
    prompt: "this is unacceptable"
    assertions:
      - type: not_contains
        label: no filler apology
        value: "apologize for any inconvenience"
"""


def _ab_run(tmp_path) -> tuple[str, str]:
    suite = tmp_path / "ab.yaml"
    suite.write_text(AB_SUITE, encoding="utf-8")
    db = str(tmp_path / "ab.db")
    result = runner.invoke(app, ["run", str(suite), "--db", db])
    assert result.exit_code == 1  # each target fails one case
    return str(suite), db


def test_compare_reports_wins_on_both_sides(tmp_path):
    _, db = _ab_run(tmp_path)
    result = runner.invoke(
        app, ["compare", "v1-live", "v2-candidate", "latest", "--db", db]
    )
    assert result.exit_code == 0  # informational unless --fail-if-worse
    assert "a_only" in result.output  # v2 broke `escalate`
    assert "b_only" in result.output  # v2 fixed `tone`
    assert "exact escalation phrase" in result.output


def test_compare_gates_promotion_with_fail_if_worse(tmp_path):
    _, db = _ab_run(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "v1-live", "v2-candidate", "latest", "--db", db, "--fail-if-worse"],
    )
    assert result.exit_code == 1
    assert "Not safe to promote" in result.output


def test_compare_is_directional(tmp_path):
    """A/B is not symmetric: which target is the baseline changes the verdict."""
    _, db = _ab_run(tmp_path)

    # v0 fails everything, so promoting it over v1 breaks cases v1 passed.
    worse = runner.invoke(
        app, ["compare", "v1-live", "v0-broken", "latest", "--db", db, "--fail-if-worse"]
    )
    assert worse.exit_code == 1

    # Reversed, v1 breaks nothing v0 passed (v0 passed nothing), so it is safe.
    better = runner.invoke(
        app, ["compare", "v0-broken", "v1-live", "latest", "--db", db, "--fail-if-worse"]
    )
    assert better.exit_code == 0
    assert "strict improvement" in better.output


def test_compare_counts_both_directions_when_each_breaks_the_other(tmp_path):
    # v1 and v2 each pass one case the other fails, so neither is promotable.
    _, db = _ab_run(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "v2-candidate", "v1-live", "latest", "--db", db, "--fail-if-worse"],
    )
    assert result.exit_code == 1
    assert "a_only" in result.output and "b_only" in result.output


def test_compare_writes_a_self_contained_html_report(tmp_path):
    _, db = _ab_run(tmp_path)
    out = tmp_path / "ab.html"
    result = runner.invoke(
        app, ["compare", "v1-live", "v2-candidate", "latest", "--db", db, "--html", str(out)]
    )
    assert result.exit_code == 0
    html = out.read_text(encoding="utf-8")
    assert "A/B: v1-live vs v2-candidate" in html
    assert "Not safe to promote" in html
    assert "exact escalation phrase" in html
    # Still self-contained: no external assets.
    assert "http://" not in html and "https://" not in html


def test_compare_writes_machine_readable_json(tmp_path):
    import json

    _, db = _ab_run(tmp_path)
    out = tmp_path / "ab.json"
    result = runner.invoke(
        app, ["compare", "v1-live", "v2-candidate", "latest", "--db", db, "--json", str(out)]
    )
    assert result.exit_code == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["target_a"] == "v1-live"
    assert data["target_b"] == "v2-candidate"
    # The field CI actually gates on.
    assert data["b_is_worse"] is True
    assert data["counts"]["only_a_passes"] == 1
    assert data["counts"]["only_b_passes"] == 1
    verdicts = {c["case_id"]: c["verdict"] for c in data["cases"]}
    assert verdicts == {"escalate": "a_only", "tone": "b_only"}


def test_compare_with_an_unknown_target_is_a_usage_error(tmp_path):
    _, db = _ab_run(tmp_path)
    result = runner.invoke(
        app, ["compare", "v1-live", "nope", "latest", "--db", db]
    )
    assert result.exit_code == 2
    assert "nope" in result.output


def test_fail_under_gate(tmp_path):
    db = str(tmp_path / "f.db")
    result = runner.invoke(app, ["run", DEMO, "--db", db, *DEGRADE, "--fail-under", "0.99"])
    assert result.exit_code == 1
