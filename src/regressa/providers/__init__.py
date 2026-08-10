"""Provider registry."""

from __future__ import annotations

from regressa.providers.anthropic_provider import AnthropicProvider
from regressa.providers.base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderError,
)
from regressa.providers.mock import MockProvider

_REGISTRY: dict[str, type] = {
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


def get_provider(name: str) -> Provider:
    try:
        return _REGISTRY[name]()  # type: ignore[return-value]
    except KeyError as exc:
        raise ProviderError(
            f"unknown provider {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def register(name: str, factory: type) -> None:
    """Register a custom provider (e.g. a local model or an internal gateway)."""
    _REGISTRY[name] = factory


__all__ = [
    "AnthropicProvider",
    "MockProvider",
    "CompletionRequest",
    "CompletionResponse",
    "Provider",
    "ProviderError",
    "get_provider",
    "register",
]
