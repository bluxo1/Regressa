# Regressa

[![CI](https://github.com/bluxo1/Regressa/actions/workflows/ci.yml/badge.svg)](https://github.com/bluxo1/Regressa/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#license)

**Prompt evaluation and regression-testing suite.** Run a fixed set of prompts
against an LLM, check the outputs against expected criteria, and flag
regressions when prompts, models, or system instructions change.

The problem it solves: prompt changes have no compiler and no type system. You
tweak a system instruction to fix one behaviour, ship it, and silently break
three others. Regressa gives that workflow a test suite, a baseline, and a diff.

```bash
# No API key and no cost — the mock provider exercises the whole harness.
regressa run suites/offline-demo.yaml
```

📄 **[See a live sample report](https://bluxo1.github.io/Regressa/report.html)** —
a real self-contained HTML report, regenerated from `suites/offline-demo.yaml` on
every Pages deploy, so it can't drift from what the tool actually produces. The
[project page](https://bluxo1.github.io/Regressa/) is built from `docs/`. (In a
fork, turn it on under Settings → Pages → Source: "GitHub Actions".)

## Contents

- [Install](#install) · [Quick start](#quick-start) · [Writing a suite](#writing-a-suite)
- [Detecting regressions](#detecting-regressions) · [Cost, caching, and capability checks](#cost-caching-and-capability-checks)
- [Running in CI](#running-in-ci) · [Verifying and tuning](#verifying-and-tuning)
- [CLI reference](#cli-reference) · [Offline development](#offline-development)
- [Architecture](#architecture) · [Tests](#tests) · [License](#license)

---

## Install

```bash
pip install -e .
```

Semantic-similarity assertions need a local embedding model (the Anthropic API
has no embeddings endpoint, so this can't run on your API key):

```bash
pip install -e '.[similarity]'
```

Credentials resolve exactly the way the Anthropic SDK resolves them: set
`ANTHROPIC_API_KEY`, or run `ant auth login` and leave it unset.

---

## Quick start

```bash
# 1. See what a suite would do, without calling anything.
regressa run suites/offline-demo.yaml --dry-run

# 2. Run it and record the result as the baseline.
regressa run suites/offline-demo.yaml --set-baseline

# 3. Change something, then re-run. Anything that used to pass and now fails is
#    reported as a regression, and the command exits 1 so CI catches it.
#    (The bundled demo simulates the change with --var; in real use you'd edit
#    the prompt, model, or system instruction and just re-run.)
regressa run suites/offline-demo.yaml \
  --var greeting_reply="As an AI language model, I cannot assist with that."

# 4. Look at movement over time.
regressa trend --suite offline-demo
regressa report latest -o report.html
```

`suites/support-assistant.yaml` is the realistic worked example: a customer
support prompt with explicit rules for refunds, escalation, tone and format,
carrying two versions of that prompt as two targets. The rewrite in
`v2-candidate` is a plausible improvement that quietly drops an exact phrase the
ticket router depends on — which is the whole reason this tool exists. It calls
the real API, so it needs a key.

Step 3 prints exactly what broke and why:

```
| change    | case     | target | score        | detail                                    |
| regressed | greeting | agent  | 1.00 -> 0.00 | names the product: missing ['Northwind    |
|           |          |        |              | Lamp']; no AI boilerplate: found banned   |
|           |          |        |              | ['As an AI', 'I cannot assist']           |
1 regression(s). Non-zero exit for CI.
```

---

## Writing a suite

A suite is one YAML file with three parts: **targets** (what to test),
**cases** (the fixed prompts), and **assertions** (what correct looks like).

```yaml
name: support-assistant

vars:
  refund_window_days: 30

defaults:
  max_tokens: 1024
  system: |
    You are a support agent. Refunds are available within
    {{ refund_window_days }} days of delivery. Answer in three sentences or fewer.
  # Applied to every case — global guardrails belong here, not copy-pasted.
  # Match verbatim fragments of your prompt, not talk *about* it: banning
  # "my instructions" would fail the model for correctly saying
  # "I can't share my instructions".
  assertions:
    - type: not_contains
      label: no prompt leak
      value: ["Rules you must follow", "Never reveal these instructions"]

targets:
  - id: prod                       # current production config
    model: claude-sonnet-5
    effort: low
  - id: candidate                  # the change you're evaluating
    model: claude-opus-5
    effort: low

cases:
  - id: refund-window
    tags: [policy, critical]
    prompt: "I bought a lamp three weeks ago. Can I return it?"
    assertions:
      - type: contains
        label: states the window
        value: ["30 day", "30-day"]
        mode: any
        weight: 2                  # counts double toward the case score
      - type: length
        unit: words
        max: 90
```

A **target** is the axis of change you're hunting regressions along. Swap the
model, edit the system prompt, or change `effort`, then diff against the
baseline. Cases and assertions stay fixed; that's what makes the comparison mean
something.

`{{ var }}` interpolation works in prompts and the system prompt, resolved from
suite-level `vars` merged with case-level `vars`, and overridable per run with
`--var key=value`. A missing variable is an error — rendering `None` into a
prompt would corrupt an eval without failing it.

`--var` is how you run one suite against several prompt variants without editing
the file. Because a diff between two differently-parameterised runs would
otherwise look like an unexplained regression, overrides are recorded in the run
label automatically when you don't pass `--label`.

### Two comparisons, two commands

Regressa answers two different questions, and it matters which one you're asking:

| Question | Command | Shape |
|---|---|---|
| Did today break what worked yesterday? | `regressa diff` | One target, two runs |
| Is version B better than version A? | `regressa compare` | Two targets, one run |

`diff` is keyed on the `(case, target)` pair, so comparing two *different* target
ids across runs gives you `added` / `removed` rows rather than regressions. That
is correct — they're different things — but it's not the A/B you wanted. For
that, put both versions in one suite as two targets and use `compare`: both saw
the same inputs at the same moment, so model drift and flake cancel out.

```bash
regressa run suites/support-assistant.yaml
regressa compare v1-live v2-candidate --fail-if-worse
```

```
| verdict | case                     | v1-live | v2-candidate | score        | detail                    |
| a_only  | escalate-security-breach |    +    |      x       | 1.00 -> 0.00 | v2-candidate failed -     |
|         |                          |         |              |              | exact escalation phrase:  |
|         |                          |         |              |              | missing ['escalating to   |
|         |                          |         |              |              | a specialist']            |
| b_only  | tone-under-pressure      |    x    |      +       | 0.00 -> 1.00 | v1-live failed - no       |
|         |                          |         |              |              | filler apology            |
v2-candidate fails 2 case(s) that v1-live passes. Not safe to promote over v1-live as-is.
```

`--fail-if-worse` exits 1 when the candidate fails anything the baseline passes,
which is the gate you want in a PR that edits a prompt. Without it the command is
informational and exits 0.

For CI, `--html` writes the A/B as a single self-contained file you can attach to
a build, and `--json` writes a machine-readable form whose `b_is_worse` field is
the one flag to gate on:

```bash
regressa compare v1-live v2-candidate --html ab.html --json ab.json --fail-if-worse
```

### Assertions

**Deterministic** — free, fast, no flake. These carry the load:

| Type | Checks |
|---|---|
| `contains` / `not_contains` | Substring presence. `mode: any` \| `all`; `contains` gives partial credit. |
| `equals` | Exact match, with optional `strip` / `ignore_case`. |
| `regex` | Pattern match; `expect: false` asserts it must *not* match. |
| `starts_with` / `ends_with` | Prefix / suffix. |
| `json_valid` | Output parses as JSON (tolerates a ```json fence). |
| `json_schema` | Validates against a JSON Schema; partial credit by error count. |
| `json_path` | Dotted path lookup (`items.0.sku`), with `equals` or `exists`. |
| `length` | Size in `chars` \| `words` \| `lines` \| `sentences`, with `min` / `max`. |
| `latency` | Wall-clock budget in ms. Skipped on cache hits. |
| `cost` | Per-call USD budget, computed from real token usage. |
| `tokens` | `max_input` / `max_output` ceilings. |

**Model-graded** — costs money, not perfectly deterministic:

| Type | Checks |
|---|---|
| `llm_judge` | Grades against a `rubric` with a grader model; `threshold` sets the pass bar. Uses structured outputs, so the verdict is parsed from schema-constrained JSON rather than scraped from prose. |
| `similarity` | Embedding cosine similarity against a `reference`. Needs the `similarity` extra. **Measures topic, not truth** — "refunds within 30 days" and "we never offer refunds" score alike, so don't use it to assert policy. Use `contains` or `llm_judge` for that. |

Every assertion takes an optional `label` (shown in reports) and a `weight`
(default 1). A case **passes** only if every assertion passes; its **score** is
the weighted mean of assertion scores. That split matters: pass/fail is the CI
gate, while the score is the continuous signal that shows a prompt degrading
before it actually breaks.

---

## Detecting regressions

Every run is persisted to SQLite (`.regressa/regressa.db`). One run is pinned as
the **baseline**, and each new run is diffed against it.

```bash
regressa baseline set latest --suite support-assistant
regressa baseline show
regressa diff baseline latest --suite support-assistant
regressa diff latest~1 latest --suite support-assistant
```

The diff classifies every case/target pair:

| Class | Meaning | Fails the build |
|---|---|---|
| `regressed` | Passed before, fails now. | **yes** |
| `score_drop` | Score fell by more than `--tolerance` (default 0.05). | **yes** |
| `errored` | Provider or network failure. | no |
| `fixed` / `score_gain` | Improvement. | no |
| `added` / `removed` | Case/target pair appeared or disappeared. | no |
| `unchanged` | Within tolerance. | no |

Two of those classifications are deliberate, and they're the reason the gate is
worth trusting:

- **A provider error is not a regression.** A network blip is a fact about your
  connection, not your prompt. Recording it as a regression trains people to
  ignore red builds.
- **A brand-new failing case is not a regression.** It never passed, so nothing
  regressed. It shows up as `added` and you deal with it as new work.

`--tolerance` exists because judge scores jitter slightly between identical runs.
Score moves smaller than it are treated as noise.

If the two runs came from different suite definitions, the diff says so (each run
records a hash of the suite it came from) — a score change may be an edited
assertion rather than a real regression.

---

## Cost, caching, and capability checks

**Cost** is computed per call from real token usage against a per-model price
table (`regressa models`), and includes grader-model spend from `llm_judge`
assertions so nothing hides off-ledger.

Promotional pricing is **date-aware**. Sonnet 5's introductory rate ($2/$10 per
MTok versus the $3/$15 list rate) is modelled with an explicit end date, so costs
are computed at the rate actually in force rather than at whichever number
happened to be hardcoded. `regressa models` shows `promo/list` side by side and
warns how many days remain, because a promotion ending shifts every cost figure
with no prompt change — enough to trip a `cost` assertion overnight. Costs are
stored on the run when it happens, so historical runs keep the price they were
actually charged.

**Caching** is on by default. Responses are keyed on a hash of everything that
affects output — model, prompt, system, max_tokens, effort, thinking, stop
sequences — so re-running a suite after editing one case only pays for that
case. `--no-cache` forces fresh calls; `regressa cache-clear` empties it.

**Capability validation** runs before any request is made. The Claude API isn't
uniform across models, and an eval harness is exactly the tool people point at
several models at once:

- `temperature` / `top_p` are **rejected with a 400** on Opus 5, Sonnet 5, and
  Fable 5. A target that sets them is refused up front with an explanation,
  rather than 400-ing on every case mid-run.
- `thinking: disabled` is only accepted at effort `high` or below on Opus 5, and
  not at all on Fable 5.
- `max_tokens` above a model's ceiling is caught before the call.

```bash
regressa validate suites/          # schema + capability check, no calls
regressa models                    # pricing, context, sampling support
```

Requests above ~16k `max_tokens` stream automatically, because the SDK refuses
non-streaming requests it estimates will exceed its HTTP timeout.

---

## Running in CI

```yaml
- run: pip install -e .
- run: regressa run suites/ --json results.json --html report.html
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Exit codes are the contract:

| Code | Meaning |
|---|---|
| `0` | Everything passed, no regressions against the baseline. |
| `1` | Assertion failures, regressions, or below `--fail-under`. |
| `2` | Configuration or usage error (bad suite, uncallable target). |

Commit `.regressa/regressa.db` to keep history and the baseline across CI runs,
or cache it as a CI artifact. The HTML report is a single self-contained file
with no external assets, so it attaches to a build cleanly.

`.github/workflows/ci.yml` ships a working setup with three jobs:

| Job | Trigger | Needs a key |
|---|---|---|
| `test` | every push and PR | no |
| `similarity` | manual / scheduled | no (installs torch, so it's kept off the hot path) |
| `eval` | manual / scheduled | optional — skips without `ANTHROPIC_API_KEY` |

The `test` job runs `pytest -m "not slow"`, validates the bundled suites, and
asserts that a known regression *does* fail the build — a gate nobody has
verified is a gate you shouldn't trust. The `eval` job runs `regressa doctor`
first, so a broken integration costs one cheap call instead of a whole suite.

Without the secret, `eval` skips its steps and the run stays green: a clone with
no key configured shouldn't sit permanently red on a schedule it was never
going to be able to run. Add `ANTHROPIC_API_KEY` to the repository secrets to
turn it on.

---

## Verifying and tuning

Two commands exist because two things can't be checked offline.

**`regressa doctor`** makes a handful of tiny calls to confirm the integration
actually works: requests accepted, responses parsed, usage populated, cost
non-zero, the streaming branch, and the judge's structured output. One check is
free and the most valuable — it sends `temperature` to a model the capability
table says rejects it, and *fails if the API accepts it*, which would mean
Regressa is refusing valid target configurations for no reason.

```bash
regressa doctor                          # cheapest model, ~$0.001
regressa doctor --model claude-sonnet-5  # also exercises the effort parameter
```

**`regressa calibrate`** sizes `--tolerance` from runs you already have, so it
costs nothing. It separates two problems that look alike:

- **Score jitter** — the same case scoring 0.82 then 0.79. A tolerance absorbs it.
- **Flapping** — the same case passing then failing. No tolerance fixes this,
  because pass/fail isn't a threshold on the score. Flapping cases are excluded
  from the recommendation and reported separately: widening the gate to cover one
  would mask real regressions everywhere else.

```bash
regressa calibrate --suite support-assistant
```

For a meaningful number the runs must be real and made with `--no-cache` —
cached responses are byte-identical by construction, so they measure nothing.

---

## CLI reference

| Command | What it does |
|---|---|
| `regressa run [PATHS]` | Run suites. `--target` `--tag` `--case` filter; `--var k=v` overrides variables; `-j` sets concurrency; `--dry-run` prints the plan. |
| `regressa validate [PATHS]` | Schema and capability check, no model calls. |
| `regressa diff BEFORE [AFTER]` | Compare one target across two runs. Accepts a run id, `baseline`, `latest`, `latest~N`. |
| `regressa compare A B [RUN]` | A/B two targets inside one run. `--fail-if-worse` gates promotion; `--html` / `--json` write artifacts. |
| `regressa baseline set/show/clear` | Manage the comparison baseline. |
| `regressa history` | List recorded runs, newest first. `--suite` filters to one suite; `-n` caps the number of runs (default 20). |
| `regressa trend` | Score movement across runs. `--suite` narrows to one suite, otherwise every suite with history is shown; `-n` caps the number of runs (default 30). |
| `regressa show [RUN]` | Print a stored run. |
| `regressa report [RUN]` | Write an HTML report. |
| `regressa doctor` | Verify the live Anthropic integration. Needs a key; costs a fraction of a cent. |
| `regressa calibrate` | Measure run-to-run jitter from history and recommend a `--tolerance`. Free. |
| `regressa models` | Known models: pricing, limits, sampling support. |
| `regressa cache-clear` | Drop cached responses. |

---

## Offline development

The `mock` provider returns deterministic responses from target `options`, so the
entire harness — runner, assertions, storage, baselines, diffing, reports — runs
with no API key and no cost. It's how the test suite works, and it's the fastest
way to iterate on assertions before spending anything.

```yaml
targets:
  - id: fixture
    provider: mock
    model: mock-1
    options:
      default: "Refunds are available within 30 days."
      by_case:
        greeting: "Hello! How can I help?"
      rules:
        - contains: "invoice"          # substring match on the rendered prompt
          response: "Invoices go out monthly."
      fail: null                       # set a string to simulate a provider error
```

Mock responses are themselves templated, so `--var` can swap a fixture without
editing the suite — that's how the bundled demo simulates a behaviour change
while keeping the target id, and therefore the diff key, stable.

---

## Architecture

```
src/regressa/
  capabilities.py     per-model pricing + what each model actually accepts
  config.py           suite schema (Pydantic; unknown keys rejected)
  loader.py           YAML loading and suite discovery
  templating.py       {{ var }} interpolation
  providers/          anthropic (capability-aware) + mock (offline)
  assertions/         deterministic / judge / similarity evaluators
  runner.py           plan → call → evaluate, with concurrency and caching
  storage.py          SQLite runs, results, baselines, cache, trends
  diffing.py          run-to-run comparison and regression classification
  reporting/          rich console + self-contained HTML
  cli.py              Typer CLI
```

Adding an assertion type means a Pydantic spec in `config.py` (added to the
`AssertionSpec` union) plus an `@evaluator("your_type")` function. Adding a
provider means one class with a `complete()` method, registered in
`providers/__init__.py` — useful for a local model or an internal gateway.

---

## Tests

```bash
pytest                  # everything installed
pytest -m "not slow"    # skip the tests that need a local embedding model
```

Everything runs offline against the mock provider and stubbed clients — no API
key, no network, no cost.

Tests marked `slow` exercise the `similarity` assertion against a real
`sentence-transformers` model. They **skip automatically** when the extra isn't
installed, so a plain `pytest` is green either way; install
`pip install -e '.[similarity]'` to actually run them.

## License

MIT
