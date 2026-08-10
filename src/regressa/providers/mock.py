"""Deterministic offline provider.

Its purpose is not to fake a model but to make the *harness* testable: every
runner path, assertion, storage write, and diff can be exercised with no API key
and no cost. It is also useful for pinning fixture outputs while you iterate on
assertions.

Configure it through a target's `options`:

    targets:
      - id: fixture
        provider: mock
        model: mock-1
        options:
          latency_ms: 5
          default: "Refunds are available within 30 days."
          by_case:
            refund-policy: "Our refund window is 30 days from delivery."
          rules:
            - contains: "invoice"       # substring match on the rendered prompt
              response: "Invoices are emailed monthly."
            - regex: "order [A-Z]-\\d+"
              response: '{"status": "shipped"}'
"""

from __future__ import annotations

import re
import time
from typing import Any

from regressa import templating
from regressa.providers.base import CompletionRequest, CompletionResponse, ProviderError


class MockProvider:
    name = "mock"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        options = request.options or {}
        started = time.perf_counter()

        latency_ms = float(options.get("latency_ms", 0) or 0)
        if latency_ms:
            time.sleep(latency_ms / 1000)

        if options.get("fail"):
            raise ProviderError(str(options["fail"]))

        text = self._resolve(request, options)
        # Mock-only convenience: responses go through the same `{{ var }}`
        # templating as prompts, so one fixture suite can be parameterised with
        # `regressa run --var reply=...` to simulate a behaviour change while
        # keeping the target id — and therefore the diff key — stable.
        if "{{" in text:
            text = templating.render(text, request.variables)
        elapsed = (time.perf_counter() - started) * 1000

        return CompletionResponse(
            text=text,
            # Word-count stand-ins so token assertions have something to bite on.
            input_tokens=_approx_tokens(request.system or "") + _approx_tokens(request.prompt),
            output_tokens=_approx_tokens(text),
            latency_ms=elapsed,
            stop_reason="end_turn",
        )

    def _resolve(self, request: CompletionRequest, options: dict[str, Any]) -> str:
        by_case = options.get("by_case") or {}
        if request.case_id and request.case_id in by_case:
            return str(by_case[request.case_id])

        for rule in options.get("rules") or []:
            if "contains" in rule and str(rule["contains"]).lower() in request.prompt.lower():
                return str(rule["response"])
            if "regex" in rule and re.search(str(rule["regex"]), request.prompt):
                return str(rule["response"])

        if options.get("echo"):
            return request.prompt

        if "default" in options:
            return str(options["default"])

        raise ProviderError(
            f"mock provider has no response for case {request.case_id!r}: set "
            "`options.default`, `options.by_case`, `options.rules`, or `options.echo`"
        )


def _approx_tokens(text: str) -> int:
    """Rough stand-in for a tokenizer.

    Never use this for real cost estimates — count tokens with
    `client.messages.count_tokens` against the actual model. It exists only so
    the mock provider populates usage fields with plausible numbers.
    """
    return max(1, len(text) // 4) if text else 0
