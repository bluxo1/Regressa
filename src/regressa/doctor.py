"""Verify the Anthropic integration against the live API.

Everything else in Regressa is tested offline with stub clients, which is fast and
free but cannot tell you whether the requests it builds are actually accepted, or
whether the responses it parses look like real ones. This module makes a handful
of deliberately tiny calls to answer that.

It costs a fraction of a cent on the default model, and one check costs nothing at
all: a request the API is expected to *reject*, which validates the capability
table the rest of the tool depends on. If that check ever passes, Regressa is
refusing target configurations the API would actually accept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from regressa import capabilities
from regressa.assertions import judge
from regressa.providers.anthropic_provider import (
    LOW_CREDIT_HINT,
    NO_CREDENTIALS_HINT,
    STREAMING_THRESHOLD,
    AnthropicProvider,
)
from regressa.providers.base import CompletionRequest, ProviderError

# Not the cheapest model, deliberately. Haiku 4.5 accepts sampling parameters and
# has no `effort`, so defaulting to it skipped the two checks that matter most --
# including the capability-table check, which is the main reason this exists. The
# calls here are ~25 tokens each, so the difference is a fraction of a cent, and
# Sonnet 5 is also what the bundled suite targets. Override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# Short prompt, short answer: these calls exist to test plumbing, not the model.
PING = "Reply with exactly: OK"


@dataclass
class Check:
    name: str
    # True = verified, False = broken, None = not applicable to this model.
    passed: bool | None
    detail: str
    cost_usd: float = 0.0

    @property
    def skipped(self) -> bool:
        return self.passed is None


@dataclass
class Diagnosis:
    model: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.passed is False]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.checks)


def _check(name: str):
    """Wrap a check so an unexpected exception is a reported failure, not a crash."""

    def decorate(fn):
        def wrapper(*args: Any, **kwargs: Any) -> Check:
            try:
                return fn(*args, **kwargs)
            except ProviderError as exc:
                return Check(name, False, str(exc))
            except Exception as exc:  # noqa: BLE001
                return Check(name, False, f"{type(exc).__name__}: {exc}")

        return wrapper

    return decorate


@_check("anthropic sdk")
def check_sdk(provider: AnthropicProvider) -> Check:
    """The package imports and a client can be constructed.

    This is not an auth check: the SDK resolves credentials lazily at request
    time, so a client constructs fine with no key at all. Auth is verified by the
    first real request below.
    """
    provider.client  # noqa: B018 - property construction is the check
    return Check("anthropic sdk", True, "package importable; client constructed")


def check_basic_request(
    provider: AnthropicProvider, model: str
) -> tuple[Check, Any | None]:
    """The core path: a request Regressa built, parsed back into a response.

    Returns the response too, so later checks can reuse it instead of paying for
    another identical call.
    """
    name = "request accepted"
    try:
        response = provider.complete(
            CompletionRequest(model=model, prompt=PING, max_tokens=16)
        )
    except ProviderError as exc:
        return Check(name, False, str(exc)), None
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"{type(exc).__name__}: {exc}"), None

    problems = []
    if not response.text.strip():
        problems.append("response text was empty")
    if response.input_tokens <= 0:
        problems.append("usage.input_tokens was not populated")
    if response.output_tokens <= 0:
        problems.append("usage.output_tokens was not populated")
    if response.stop_reason is None:
        problems.append("stop_reason was not populated")

    cost = capabilities.price(model, response.input_tokens, response.output_tokens)
    if problems:
        return Check(name, False, "; ".join(problems), cost), response
    return (
        Check(
            name,
            True,
            f"in={response.input_tokens} out={response.output_tokens} "
            f"stop={response.stop_reason} text={response.text.strip()[:40]!r}",
            cost,
        ),
        response,
    )


def check_cost(model: str, response: Any) -> Check:
    """Real usage must produce a non-zero cost, or every budget is meaningless.

    Reuses the response from the request check rather than spending another call.
    """
    name = "cost accounting"
    cost = capabilities.price(model, response.input_tokens, response.output_tokens)
    rates = capabilities.get(model).rates()
    if cost <= 0:
        return Check(name, False, "computed cost was zero for real usage")
    note = " (promotional rate in force)" if rates.promo_active else ""
    return Check(
        name,
        True,
        f"${cost:.6f} at ${rates.input_per_mtok:.2f}/${rates.output_per_mtok:.2f} "
        f"per MTok{note}",
        # Cost already attributed to the request check; don't double-count.
        0.0,
    )


@_check("effort parameter")
def check_effort(provider: AnthropicProvider, model: str) -> Check:
    """`output_config.effort` is accepted where the capability table claims it is."""
    caps = capabilities.get(model)
    if not caps.effort_levels:
        return Check(
            "effort parameter", None, f"{model} has no effort parameter; nothing to check"
        )
    response = provider.complete(
        CompletionRequest(model=model, prompt=PING, max_tokens=16, effort="low")
    )
    cost = capabilities.price(model, response.input_tokens, response.output_tokens)
    return Check("effort parameter", True, "effort=low accepted", cost)


@_check("streaming path")
def check_streaming(provider: AnthropicProvider, model: str) -> Check:
    """Large max_tokens must take the streaming branch and still parse.

    max_tokens is a ceiling, not a target, so this stays cheap: the prompt asks
    for two characters.
    """
    caps = capabilities.get(model)
    budget = min(STREAMING_THRESHOLD + 1_000, caps.max_output_tokens)
    if budget <= STREAMING_THRESHOLD:
        return Check(
            "streaming path", None, f"{model} cannot exceed the streaming threshold"
        )
    response = provider.complete(
        CompletionRequest(model=model, prompt=PING, max_tokens=budget)
    )
    cost = capabilities.price(model, response.input_tokens, response.output_tokens)
    if not response.text.strip():
        return Check("streaming path", False, "streamed response had no text", cost)
    return Check(
        "streaming path", True, f"max_tokens={budget} streamed and parsed", cost
    )


@_check("structured output (judge)")
def check_structured_output(model: str) -> Check:
    """The judge relies on schema-constrained JSON; verify the API honours it."""
    client = judge._get_client()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=judge.JUDGE_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": judge._SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": (
                    "<rubric>The reply must say the word yes.</rubric>\n\n"
                    "<prompt_given_to_model>say yes</prompt_given_to_model>\n\n"
                    "<candidate_output>yes</candidate_output>"
                ),
            }
        ],
    )
    usage = getattr(message, "usage", None)
    cost = capabilities.price(
        model,
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )

    verdict = judge._parse(message)
    if verdict is None:
        return Check(
            "structured output (judge)", False, "grader output was not parseable JSON", cost
        )
    if "score" not in verdict or "reasoning" not in verdict:
        return Check(
            "structured output (judge)",
            False,
            f"grader JSON missing required keys: {json.dumps(verdict)[:120]}",
            cost,
        )
    return Check(
        "structured output (judge)",
        True,
        f"schema honoured; score={verdict['score']}",
        cost,
    )


def check_sampling_is_rejected(model: str) -> Check:
    """Confirm the API really rejects `temperature` on models we gate it on.

    This is the single most consequential assumption in `capabilities.py`:
    Regressa refuses to send sampling parameters to these models. If the API in
    fact accepts them, that refusal is wrong and users are being blocked for no
    reason. A rejected request is not billed, so this check is free.
    """
    name = "sampling correctly rejected"
    caps = capabilities.get(model)
    if caps.supports_sampling:
        return Check(name, None, f"{model} accepts sampling parameters; nothing to check")

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        return Check(name, False, f"anthropic package unavailable: {exc}")

    client = anthropic.Anthropic()
    try:
        client.messages.create(
            model=model,
            max_tokens=8,
            temperature=0.5,
            messages=[{"role": "user", "content": PING}],
        )
    except anthropic.BadRequestError as exc:
        return Check(name, True, f"API rejected temperature as expected ({exc.status_code})")
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"unexpected error instead of a 400: {type(exc).__name__}: {exc}")

    return Check(
        name,
        False,
        f"{model} ACCEPTED temperature - the capability table is wrong and Regressa "
        "is refusing valid targets. Set supports_sampling=True for this model.",
    )


def diagnose(model: str = DEFAULT_MODEL, provider: AnthropicProvider | None = None
             ) -> Diagnosis:
    """Run every check against `model`.

    Bails out after the first request if credentials are missing. Every later
    check would fail for the same reason, and six copies of one auth error buries
    the single line the user needs to act on.
    """
    provider = provider or AnthropicProvider()
    result = Diagnosis(model=model)

    sdk = check_sdk(provider)
    result.checks.append(sdk)
    if sdk.passed is False:
        return result

    basic, response = check_basic_request(provider, model)
    # Two account-level problems make every later check fail for the same reason.
    # Report each once and stop, rather than repeating one error six times.
    if basic.passed is False:
        if NO_CREDENTIALS_HINT in basic.detail:
            result.checks.append(Check("credentials", False, NO_CREDENTIALS_HINT))
            return result
        if LOW_CREDIT_HINT in basic.detail:
            result.checks.append(Check("account balance", False, LOW_CREDIT_HINT))
            return result
    result.checks.append(basic)

    if response is not None:
        result.checks.append(check_cost(model, response))

    result.checks.append(check_sampling_is_rejected(model))  # free: expects a 400
    result.checks.append(check_effort(provider, model))
    result.checks.append(check_streaming(provider, model))
    result.checks.append(check_structured_output(model))
    return result
