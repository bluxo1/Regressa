"""Provider interface.

A provider turns a `CompletionRequest` into a `CompletionResponse`. Keeping this
narrow is what lets the offline `mock` provider exercise the entire harness —
runner, assertions, storage, diffing — with no network and no API key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    prompt: str
    max_tokens: int
    system: str | None = None
    effort: str | None = None
    thinking: str | None = None
    stop_sequences: tuple[str, ...] | None = None
    temperature: float | None = None
    top_p: float | None = None
    options: dict[str, Any] = field(default_factory=dict, compare=False)
    # Set by the runner so mock rules can key off the case being run.
    case_id: str | None = None

    def cache_key(self) -> str:
        """Hash of everything that affects the model's output.

        `options` and `case_id` are excluded: they steer the mock provider, not a
        real model's response, and including them would fragment the cache.
        """
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "system": self.system,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "thinking": self.thinking,
            "stop_sequences": list(self.stop_sequences or ()),
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class CompletionResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionResponse:
        return cls(**data)


class ProviderError(Exception):
    """A request failed in a way that should mark the case errored, not failed.

    The distinction matters: a failed assertion is a signal about the prompt; a
    provider error is a signal about the harness or the network, and must never
    be recorded as a regression.
    """


class Provider(Protocol):
    name: str

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
