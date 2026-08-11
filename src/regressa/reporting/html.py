"""Self-contained HTML report.

No external CSS or JS — the file is a single artifact you can attach to a CI run
or email to someone. The trend chart is inline SVG for the same reason.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from regressa.diffing import RunDiff, TargetComparison, Verdict
from regressa.results import RunSummary

_CSS = """
:root {
  --bg: #ffffff; --fg: #16181d; --muted: #6b7280; --line: #e5e7eb;
  --pass: #15803d; --fail: #b91c1c; --warn: #b45309; --accent: #1d4ed8;
  --card: #f9fafb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa1ab; --line: #262b33;
    --pass: #4ade80; --fail: #f87171; --warn: #fbbf24; --accent: #7aa2ff;
    --card: #161a21;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2.25rem 0 .75rem; }
.sub { color: var(--muted); margin: 0 0 1.5rem; font-size: .9rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .75rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: .85rem 1rem; }
.card .k { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; }
.card .v { font-size: 1.5rem; font-weight: 650; margin-top: .15rem; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: .875rem; min-width: 720px; }
th, td { text-align: left; padding: .55rem .7rem; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: var(--card); font-weight: 600; position: sticky; top: 0; }
tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .72rem; font-weight: 650; }
.pass { background: color-mix(in srgb, var(--pass) 18%, transparent); color: var(--pass); }
.fail { background: color-mix(in srgb, var(--fail) 18%, transparent); color: var(--fail); }
.warn { background: color-mix(in srgb, var(--warn) 20%, transparent); color: var(--warn); }
pre { margin: .4rem 0 0; padding: .6rem .7rem; background: var(--card); border: 1px solid var(--line);
      border-radius: 8px; white-space: pre-wrap; word-break: break-word; font-size: .8rem; max-height: 18rem;
      overflow: auto; }
.note { background: color-mix(in srgb, var(--warn) 12%, transparent); border-left: 3px solid var(--warn);
        padding: .6rem .8rem; border-radius: 6px; margin: 1rem 0; font-size: .875rem; }
.assert { color: var(--muted); font-size: .8rem; }
footer { color: var(--muted); font-size: .78rem; margin-top: 3rem; }
svg { max-width: 100%; height: auto; display: block; }
"""


def render_report(
    run: RunSummary,
    trend: list[dict[str, Any]] | None = None,
    diff: RunDiff | None = None,
    comparison: TargetComparison | None = None,
) -> str:
    e = html.escape
    status_class = "pass" if run.failed == 0 else "fail"

    cards = [
        ("pass rate", f"{run.pass_rate:.0%}", status_class),
        ("passed", f"{run.passed}/{run.total}", status_class),
        ("mean score", f"{run.score:.3f}", ""),
        ("cost", f"${run.cost_usd:.4f}", ""),
        ("harness errors", str(run.errors), "warn" if run.errors else ""),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{e(k)}</div>'
        f'<div class="v">{e(v)}</div></div>'
        for k, v, _ in cards
    )

    rows = []
    for r in run.results:
        if r.errored:
            pill = '<span class="pill warn">ERROR</span>'
        elif r.passed:
            pill = '<span class="pill pass">PASS</span>'
        else:
            pill = '<span class="pill fail">FAIL</span>'

        assertions = "".join(
            f'<div class="assert">{"✓" if a.passed else "✗"} '
            f"{e(a.label or a.type)} — {e(a.message)}</div>"
            for a in r.assertions
        )
        body = assertions
        if r.error:
            body += f"<pre>{e(r.error)}</pre>"
        elif not r.passed and r.output:
            body += f"<pre>{e(r.output[:4000])}</pre>"

        rows.append(
            "<tr>"
            f"<td>{pill}</td>"
            f"<td><code>{e(r.case_id)}</code></td>"
            f"<td><code>{e(r.target_id)}</code></td>"
            f'<td class="num">{r.score:.2f}</td>'
            f'<td class="num">{"cached" if r.cached else f"{r.latency_ms:.0f}ms"}</td>'
            f'<td class="num">${r.usage.cost_usd:.5f}</td>'
            f"<td>{body}</td>"
            "</tr>"
        )

    sections = [
        f"<h1>{e(run.suite)}</h1>",
        f'<p class="sub">run <code>{e(run.run_id)}</code> · '
        f"{e(run.started_at.strftime('%Y-%m-%d %H:%M UTC'))}"
        + (f" · git <code>{e(run.git_sha)}</code>" if run.git_sha else "")
        + (f" · label <code>{e(run.label)}</code>" if run.label else "")
        + f" · suite hash <code>{e(run.suite_hash)}</code></p>",
        f'<div class="cards">{card_html}</div>',
    ]

    # The A/B goes above the run's own results: when you asked for a comparison,
    # the verdict is the headline and the per-case table is the evidence.
    if comparison is not None:
        sections.append(_comparison_section(comparison))

    if diff is not None:
        sections.append(_diff_section(diff))

    if trend:
        sections.append("<h2>Score trend</h2>")
        sections.append(_trend_svg(trend))

    sections.append("<h2>Results</h2>")
    sections.append(
        '<div class="scroll"><table><thead><tr>'
        "<th></th><th>case</th><th>target</th><th>score</th><th>latency</th>"
        "<th>cost</th><th>assertions</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    sections.append(
        "<footer>Generated by Regressa · "
        f"{e(datetime.now().strftime('%Y-%m-%d %H:%M'))}</footer>"
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Regressa — {e(run.suite)}</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(sections)}</main></body></html>"
    )


def _diff_section(diff: RunDiff) -> str:
    e = html.escape
    out = ["<h2>Changes vs baseline</h2>"]
    if diff.suite_changed:
        out.append(
            '<div class="note">These runs came from different suite definitions '
            "(suite hash differs), so some differences may be edits to the suite "
            "rather than model or prompt regressions.</div>"
        )
    interesting = diff.regressions + diff.fixes + diff.errors
    if not interesting:
        out.append("<p>No regressions, fixes, or errors versus the baseline.</p>")
        return "".join(out)

    rows = []
    for d in interesting:
        cls = "fail" if d.is_regression else "warn" if d.change.value == "errored" else "pass"
        before = "—" if d.before_score is None else f"{d.before_score:.2f}"
        after = "—" if d.after_score is None else f"{d.after_score:.2f}"
        rows.append(
            "<tr>"
            f'<td><span class="pill {cls}">{e(d.change.value)}</span></td>'
            f"<td><code>{e(d.case_id)}</code></td>"
            f"<td><code>{e(d.target_id)}</code></td>"
            f'<td class="num">{before} → {after}</td>'
            f"<td>{e(d.detail)}</td>"
            "</tr>"
        )
    out.append(
        '<div class="scroll"><table><thead><tr><th>change</th><th>case</th>'
        "<th>target</th><th>score</th><th>detail</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return "".join(out)


def _trend_svg(points: list[dict[str, Any]], width: int = 900, height: int = 220) -> str:
    """Inline SVG line chart of mean score over runs.

    Hand-rolled rather than pulled from a chart library so the report stays a
    single self-contained file with no network dependency.
    """
    if len(points) < 2:
        return "<p class='sub'>Need at least two runs to draw a trend.</p>"

    pad_l, pad_r, pad_t, pad_b = 44, 12, 14, 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(points)

    def x(i: int) -> float:
        return pad_l + (plot_w * i / (n - 1))

    def y(score: float) -> float:
        return pad_t + plot_h * (1 - max(0.0, min(1.0, score)))

    line = " ".join(f"{x(i):.1f},{y(p['score']):.1f}" for i, p in enumerate(points))
    area = f"{pad_l},{pad_t + plot_h} {line} {x(n - 1):.1f},{pad_t + plot_h}"

    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{y(v):.1f}" x2="{width - pad_r}" y2="{y(v):.1f}" '
        f'stroke="currentColor" stroke-opacity=".12"/>'
        f'<text x="{pad_l - 8}" y="{y(v) + 4:.1f}" text-anchor="end" '
        f'font-size="10" fill="currentColor" opacity=".55">{v:.1f}</text>'
        for v in (0.0, 0.5, 1.0)
    )

    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(p["score"]):.1f}" r="3" fill="currentColor">'
        f'<title>{html.escape(p["run_id"])} — score {p["score"]:.3f}, '
        f'{p["passed"]}/{p["total"]} passed</title></circle>'
        for i, p in enumerate(points)
    )

    first = html.escape(points[0]["started_at"][:10])
    last = html.escape(points[-1]["started_at"][:10])

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Mean score across the last {n} runs" '
        'style="color: var(--accent)">'
        f"{gridlines}"
        f'<polygon points="{area}" fill="currentColor" fill-opacity=".10"/>'
        f'<polyline points="{line}" fill="none" stroke="currentColor" stroke-width="2"/>'
        f"{dots}"
        f'<text x="{pad_l}" y="{height - 8}" font-size="10" fill="currentColor" '
        f'opacity=".55">{first}</text>'
        f'<text x="{width - pad_r}" y="{height - 8}" text-anchor="end" font-size="10" '
        f'fill="currentColor" opacity=".55">{last}</text>'
        "</svg>"
    )


def write_report(
    path: str | Path,
    run: RunSummary,
    trend: list[dict[str, Any]] | None = None,
    diff: RunDiff | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(run, trend=trend, diff=diff), encoding="utf-8")
    return out


def write_json(path: str | Path, run: RunSummary) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return out
