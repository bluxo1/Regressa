"""Per-model capability and pricing table.

Why this exists: the Claude API is not uniform across models, and an eval harness
is exactly the tool people point at several models at once. Two differences bite:

* `temperature` / `top_p` / `top_k` are **rejected with a 400** on Opus 5,
  Sonnet 5, and Fable 5. A suite that sets temperature would fail on the newest
  models for a reason that has nothing to do with the prompt under test.
* `thinking: {"type": "disabled"}` is only accepted at effort `high` or below on
  Opus 5; pairing it with `xhigh`/`max` is a 400.

Regressa validates a target against this table *before* spending a request, so a
config mistake surfaces as a config error rather than a failed eval.

Prices are USD per million tokens, first-party Anthropic API rates, cached
2026-06-24. Bedrock/Vertex are partner-operated with separate pricing — if you
point Regressa at those, override `input`/`output` in your suite.

Promotional pricing is modelled with an explicit end date rather than baked into
the headline rate. Writing an intro price directly into `input_per_mtok` is wrong
the day the promotion ends, and silently so: every cost figure and every `cost`
assertion drifts with no code change to blame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

EffortLevel = str  # "low" | "medium" | "high" | "xhigh" | "max"

ALL_EFFORT: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class PromoPrice:
    """A temporary rate that applies up to and including `until`."""

    input_per_mtok: float
    output_per_mtok: float
    until: date


@dataclass(frozen=True)
class Rates:
    """The rates in force on a given day, list or promotional."""

    input_per_mtok: float
    output_per_mtok: float
    promo_active: bool = False
    promo_until: date | None = None

    # Cache pricing follows the standard multipliers on whichever input rate
    # is actually in force.
    @property
    def cache_read_per_mtok(self) -> float:
        return self.input_per_mtok * 0.1

    @property
    def cache_write_per_mtok(self) -> float:
        return self.input_per_mtok * 1.25


@dataclass(frozen=True)
class ModelCapabilities:
    model: str
    # List price. Use `rates()` for what a request actually costs today.
    input_per_mtok: float
    output_per_mtok: float
    context_window: int
    max_output_tokens: int
    # False for models that reject temperature/top_p/top_k with a 400.
    supports_sampling: bool = False
    # Effort levels accepted by `output_config.effort`. Empty = parameter unsupported.
    effort_levels: tuple[str, ...] = ALL_EFFORT
    # Whether an explicit thinking:{"type":"disabled"} is accepted at all.
    supports_disabled_thinking: bool = True
    # Highest effort level at which disabled thinking is still accepted.
    disabled_thinking_max_effort: str | None = "max"
    # True when omitting `thinking` still runs adaptive thinking (affects cost).
    thinking_on_by_default: bool = False
    # Temporary pricing that supersedes the list rate until its end date.
    promo: PromoPrice | None = None
    notes: str = ""

    def rates(self, on: date | None = None) -> Rates:
        """Rates in force on `on` (default: today)."""
        day = on or date.today()
        if self.promo is not None and day <= self.promo.until:
            return Rates(
                input_per_mtok=self.promo.input_per_mtok,
                output_per_mtok=self.promo.output_per_mtok,
                promo_active=True,
                promo_until=self.promo.until,
            )
        return Rates(
            input_per_mtok=self.input_per_mtok,
            output_per_mtok=self.output_per_mtok,
            promo_active=False,
            promo_until=self.promo.until if self.promo else None,
        )


_MODELS: dict[str, ModelCapabilities] = {
    "claude-opus-5": ModelCapabilities(
        model="claude-opus-5",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_sampling=False,
        supports_disabled_thinking=True,
        disabled_thinking_max_effort="high",
        thinking_on_by_default=True,
        notes="Thinking is on by default; disabling it is capped at effort=high.",
    ),
    "claude-opus-4-8": ModelCapabilities(
        model="claude-opus-4-8",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_sampling=False,
        thinking_on_by_default=False,
    ),
    "claude-sonnet-5": ModelCapabilities(
        model="claude-sonnet-5",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_sampling=False,
        thinking_on_by_default=True,
        promo=PromoPrice(
            input_per_mtok=2.00, output_per_mtok=10.00, until=date(2026, 8, 31)
        ),
        notes="Intro pricing $2/$10 per MTok through 2026-08-31, then $3/$15.",
    ),
    "claude-sonnet-4-6": ModelCapabilities(
        model="claude-sonnet-4-6",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_sampling=True,
        effort_levels=("low", "medium", "high", "max"),
    ),
    "claude-haiku-4-5": ModelCapabilities(
        model="claude-haiku-4-5",
        input_per_mtok=1.00,
        output_per_mtok=5.00,
        context_window=200_000,
        max_output_tokens=64_000,
        supports_sampling=True,
        effort_levels=(),
        notes="No effort parameter; use thinking budget_tokens if you need depth.",
    ),
    "claude-fable-5": ModelCapabilities(
        model="claude-fable-5",
        input_per_mtok=10.00,
        output_per_mtok=50.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_sampling=False,
        supports_disabled_thinking=False,
        disabled_thinking_max_effort=None,
        thinking_on_by_default=True,
        notes="Thinking is always on; any explicit thinking config is a 400.",
    ),
}

# Zero-cost stand-in so the offline provider flows through the same accounting path.
_MOCK = ModelCapabilities(
    model="mock",
    input_per_mtok=0.0,
    output_per_mtok=0.0,
    context_window=1_000_000,
    max_output_tokens=128_000,
    supports_sampling=True,
    effort_levels=ALL_EFFORT,
)


class UnknownModelError(KeyError):
    """Raised when a suite names a model Regressa has no capability data for."""


def get(model: str) -> ModelCapabilities:
    """Look up capabilities for a model id.

    Unknown ids raise rather than guessing: silently assuming a model accepts
    `temperature` would turn a config error into a runtime 400 mid-run.
    """
    if model in _MODELS:
        return _MODELS[model]
    if model.startswith("mock"):
        return _MOCK
    raise UnknownModelError(
        f"unknown model {model!r}. Known models: {', '.join(sorted(_MODELS))}. "
        "Add it to regressa/capabilities.py, or use a `mock-*` id for offline runs."
    )


def known_models() -> list[str]:
    return sorted(_MODELS)


def price(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    on: date | None = None,
) -> float:
    """Cost in USD for one request's token usage, at the rates in force on `on`.

    `on` defaults to today, which is what a live run wants. Costs are stored on
    the run when it happens, so a historical run keeps the price it was actually
    charged rather than being re-priced later at a different rate.
    """
    rates = get(model).rates(on)
    return (
        input_tokens * rates.input_per_mtok
        + output_tokens * rates.output_per_mtok
        + cache_read_tokens * rates.cache_read_per_mtok
        + cache_write_tokens * rates.cache_write_per_mtok
    ) / 1_000_000


def active_promos(on: date | None = None) -> list[tuple[str, Rates]]:
    """Models currently on promotional pricing, for surfacing in `regressa models`.

    Worth showing: a promotion ending shifts every cost figure with no prompt
    change, which can trip a `cost` assertion overnight.
    """
    day = on or date.today()
    out = []
    for name in known_models():
        rates = get(name).rates(day)
        if rates.promo_active:
            out.append((name, rates))
    return out


@dataclass
class ValidationIssue:
    target_id: str
    message: str


def validate_target(
    target_id: str,
    model: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    effort: str | None = None,
    thinking: str | None = None,
    max_tokens: int | None = None,
) -> list[ValidationIssue]:
    """Check a target's request parameters against the model's real surface.

    Returns a list of human-readable problems; empty means the target is callable.
    """
    issues: list[ValidationIssue] = []

    def bad(msg: str) -> None:
        issues.append(ValidationIssue(target_id=target_id, message=msg))

    try:
        caps = get(model)
    except UnknownModelError as exc:
        bad(str(exc))
        return issues

    if not caps.supports_sampling:
        for name, value in (("temperature", temperature), ("top_p", top_p)):
            if value is not None:
                bad(
                    f"{model} rejects `{name}` with a 400 - remove it. "
                    "Steer this model with the system prompt or `effort` instead."
                )

    if effort is not None:
        if not caps.effort_levels:
            bad(f"{model} does not support `effort`; remove it from the target.")
        elif effort not in caps.effort_levels:
            bad(
                f"effort={effort!r} is not valid for {model}. "
                f"Valid levels: {', '.join(caps.effort_levels)}."
            )

    if thinking == "disabled":
        if not caps.supports_disabled_thinking:
            bad(
                f"{model} does not accept an explicit thinking config "
                "(thinking is always on). Remove `thinking` from the target."
            )
        elif (
            effort is not None
            and caps.disabled_thinking_max_effort is not None
            and _effort_rank(effort) > _effort_rank(caps.disabled_thinking_max_effort)
        ):
            bad(
                f"{model} accepts thinking=disabled only at effort "
                f"{caps.disabled_thinking_max_effort!r} or lower, but effort={effort!r} "
                "is set. Raise thinking to `adaptive`, or lower effort."
            )

    if max_tokens is not None and max_tokens > caps.max_output_tokens:
        bad(
            f"max_tokens={max_tokens} exceeds {model}'s ceiling of "
            f"{caps.max_output_tokens}."
        )

    return issues


def _effort_rank(level: str) -> int:
    try:
        return ALL_EFFORT.index(level)
    except ValueError:
        return -1
