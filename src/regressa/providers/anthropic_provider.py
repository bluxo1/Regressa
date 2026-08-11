"""Anthropic (Claude API) provider.

Request construction is capability-aware — see `regressa.capabilities`. In short:

* `temperature` / `top_p` are only sent to models that accept them; on Opus 5,
  Sonnet 5, and Fable 5 they are a 400.
* `thinking` is only sent when the suite asks for it explicitly, and never to
  Fable 5, which rejects any explicit thinking config.
* Requests with a large `max_tokens` stream, because the SDK refuses
  non-streaming requests it estimates will exceed its HTTP timeout.
"""

from __future__ import annotations

import time
from typing import Any

from regressa import capabilities
from regressa.providers.base import CompletionRequest, CompletionResponse, ProviderError

# Above this output budget the SDK may reject a non-streaming request outright,
# and long generations risk an idle-connection drop. Stream instead.
STREAMING_THRESHOLD = 16_000

NO_CREDENTIALS_HINT = (
    "no Anthropic credentials found. Set ANTHROPIC_API_KEY, or run `ant auth login` "
    "(the SDK picks up the stored profile automatically). To run without an API key, "
    "use `provider: mock` targets - see suites/offline-demo.yaml."
)

LOW_CREDIT_HINT = (
    "the API key is valid but the account has no credit balance, so the API "
    "refused the request. Add credits at console.anthropic.com -> Plans & Billing, "
    "then re-run. Nothing was charged: rejected requests are not billed."
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency is required
                raise ProviderError(
                    "the `anthropic` package is required for the anthropic provider"
                ) from exc
            # Zero-arg construction resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
            # or an `ant auth login` profile — so an unset env var is not fatal.
            try:
                self._client = anthropic.Anthropic()
            except Exception as exc:  # noqa: BLE001
                # Missing credentials is the most common first-run failure, and the
                # SDK's own message doesn't say how to fix it. Say it here.
                raise ProviderError(NO_CREDENTIALS_HINT) from exc
        return self._client

    # ----------------------------------------------------------------- #

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        params = self.build_params(request)
        started = time.perf_counter()
        try:
            if request.max_tokens > STREAMING_THRESHOLD:
                with self.client.messages.stream(**params) as stream:
                    message = stream.get_final_message()
            else:
                message = self.client.messages.create(**params)
        except Exception as exc:  # noqa: BLE001 - surfaced as a harness error
            # The SDK resolves credentials lazily, at request time rather than at
            # construction, so a missing key surfaces here and not above.
            if _is_missing_credentials(exc):
                raise ProviderError(NO_CREDENTIALS_HINT) from exc
            # An exhausted balance is an account problem, not a prompt problem, and
            # the raw 400 body buries that under a wall of JSON on every case.
            if _is_low_credit(exc):
                raise ProviderError(LOW_CREDIT_HINT) from exc
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        return self._to_response(message, latency_ms)

    # ----------------------------------------------------------------- #

    def build_params(self, request: CompletionRequest) -> dict[str, Any]:
        """Assemble the Messages API request body for this target.

        Split out from `complete` so it can be unit-tested without a client —
        the capability gating here is the part most likely to regress.
        """
        caps = capabilities.get(request.model)

        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            params["system"] = request.system
        if request.stop_sequences:
            params["stop_sequences"] = list(request.stop_sequences)

        # Sampling parameters: only for models that accept them.
        if caps.supports_sampling:
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p

        if request.effort is not None and caps.effort_levels:
            params["output_config"] = {"effort": request.effort}

        # Only send `thinking` when explicitly configured. Fable 5 rejects any
        # explicit thinking config, so honour that even if a suite asks.
        if request.thinking is not None:
            if request.thinking == "adaptive":
                params["thinking"] = {"type": "adaptive"}
            elif request.thinking == "disabled" and caps.supports_disabled_thinking:
                params["thinking"] = {"type": "disabled"}

        return params

    def _to_response(self, message: Any, latency_ms: float) -> CompletionResponse:
        stop_reason = getattr(message, "stop_reason", None)

        # A refusal is a successful HTTP 200 with empty or partial content. Reading
        # content[0] unconditionally would crash the run, so check first and let
        # the suite assert on the refusal text if it wants to.
        text = _extract_text(message)
        if stop_reason == "refusal" and not text:
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            text = f"[refusal category={category}]"

        usage = getattr(message, "usage", None)
        return CompletionResponse(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )


def _is_missing_credentials(exc: Exception) -> bool:
    """Whether this failure is "no credentials" rather than a real API error.

    Matched on the message because the SDK raises a bare TypeError for this,
    which is indistinguishable from a genuine type bug by class alone.
    """
    text = str(exc).lower()
    return "could not resolve authentication" in text or (
        "expected one of api_key" in text
    )


def _is_low_credit(exc: Exception) -> bool:
    """Whether this 400 means "no credit balance" rather than a bad request.

    Matched on the message because the SDK raises a plain BadRequestError for
    both, and telling a user their request was malformed when their account is
    simply out of credit sends them debugging the wrong thing.
    """
    return "credit balance is too low" in str(exc).lower()


def _extract_text(message: Any) -> str:
    """Concatenate text blocks, skipping thinking / tool blocks."""
    parts: list[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)
