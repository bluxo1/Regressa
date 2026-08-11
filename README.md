# Regressa

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

---

## CLI reference

| Command | What it does |
|---|---|
| `regressa run [PATHS]` | Run suites. `--target` `--tag` `--case` filter; `--var k=v` overrides variables; `-j` sets concurrency; `--dry-run` prints the plan. |
| `regressa validate [PATHS]` | Schema and capability check, no model calls. |
| `regressa diff BEFORE [AFTER]` | Compare one target across two runs. Accepts a run id, `baseline`, `latest`, `latest~N`. |
| `regressa compare A B [RUN]` | A/B two targets inside one run. `--fail-if-worse` gates promotion; `--html` / `--json` write artifacts. |
| `regressa baseline set/show/clear` | Manage the comparison baseline. |
| `regressa history` | List recorded runs. |
| `regressa trend` | Score movement across runs. |
| `regressa show [RUN]` | Print a stored run. |
| `regressa report [RUN]` | Write an HTML report. |
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
