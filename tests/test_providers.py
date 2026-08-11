"""Anthropic response parsing, exercised with a stub client.

These paths are worth testing precisely because they only run against the real
API: a change to block extraction or refusal handling would otherwise surface as
a mid-run crash on a live suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from regressa.providers.anthropic_provider import AnthropicProvider
from regressa.providers.base import CompletionRequest, ProviderError


@dataclass
class Block:
    type: str
    text: str = ""
    thinking: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Message:
    content: list[Block] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = "end_turn"
    stop_details: Any = None


class StubMessages:
    def __init__(self, message: Any = None, raises: Exception | None = None) -> None:
        self.message = message
        self.raises = raises
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.create_calls.append(params)
        if self.raises:
            raise self.raises
        return self.message

    def stream(self, **params: Any):
        self.stream_calls.append(params)

        class Ctx:
            def __init__(self, message: Any) -> None:
                self._message = message

            def __enter__(self):
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def get_final_message(self):
                return self._message

        return Ctx(self.message)


class StubClient:
    def __init__(self, message: Any = None, raises: Exception | None = None) -> None:
        self.messages = StubMessages(message, raises)


def request(**kwargs: Any) -> CompletionRequest:
    kwargs.setdefault("model", "claude-opus-5")
    kwargs.setdefault("prompt", "hello")
    kwargs.setdefault("max_tokens", 100)
    return CompletionRequest(**kwargs)


def test_text_blocks_are_concatenated_and_thinking_is_skipped():
    message = Message(
        content=[
            Block(type="thinking", thinking="internal reasoning"),
            Block(type="text", text="Part one. "),
            Block(type="text", text="Part two."),
        ],
        usage=Usage(input_tokens=12, output_tokens=7),
    )
    provider = AnthropicProvider(StubClient(message))
    response = provider.complete(request())

    assert response.text == "Part one. Part two."
    assert "internal reasoning" not in response.text
    assert response.input_tokens == 12
    assert response.output_tokens == 7


def test_cache_token_fields_are_carried_through():
    message = Message(
        usage=Usage(cache_read_input_tokens=900, cache_creation_input_tokens=100),
        content=[Block(type="text", text="x")],
    )
    response = AnthropicProvider(StubClient(message)).complete(request())
    assert response.cache_read_tokens == 900
    assert response.cache_write_tokens == 100


def test_a_refusal_does_not_crash_on_empty_content():
    # A refusal is a successful HTTP 200 with empty content — indexing content[0]
    # unconditionally would take the whole run down.
    @dataclass
    class Details:
        category: str = "cyber"

    message = Message(content=[], stop_reason="refusal", stop_details=Details())
    response = AnthropicProvider(StubClient(message)).complete(request())

    assert response.stop_reason == "refusal"
    assert "refusal" in response.text
    assert "cyber" in response.text


def test_partial_output_before_a_refusal_is_preserved():
    message = Message(
        content=[Block(type="text", text="Here is what I can say.")],
        stop_reason="refusal",
    )
    response = AnthropicProvider(StubClient(message)).complete(request())
    assert response.text == "Here is what I can say."


def test_sdk_exceptions_become_provider_errors():
    provider = AnthropicProvider(StubClient(raises=RuntimeError("connection reset")))
    with pytest.raises(ProviderError) as exc:
        provider.complete(request())
    assert "connection reset" in str(exc.value)


def test_small_requests_do_not_stream():
    client = StubClient(Message(content=[Block(type="text", text="x")]))
    AnthropicProvider(client).complete(request(max_tokens=1000))
    assert len(client.messages.create_calls) == 1
    assert client.messages.stream_calls == []


def test_large_max_tokens_streams_instead():
    # The SDK refuses non-streaming requests it estimates will exceed its timeout.
    client = StubClient(Message(content=[Block(type="text", text="x")]))
    AnthropicProvider(client).complete(request(max_tokens=64_000))
    assert client.messages.create_calls == []
    assert len(client.messages.stream_calls) == 1


def test_system_and_stop_sequences_are_forwarded():
    client = StubClient(Message(content=[Block(type="text", text="x")]))
    AnthropicProvider(client).complete(
        request(system="be terse", stop_sequences=("END",))
    )
    params = client.messages.create_calls[0]
    assert params["system"] == "be terse"
    assert params["stop_sequences"] == ["END"]
    assert params["messages"] == [{"role": "user", "content": "hello"}]


def test_missing_credentials_produces_an_actionable_message():
    # The SDK resolves credentials lazily and raises a bare TypeError at request
    # time, so this must be recognised from the failing call, not construction.
    auth_error = TypeError(
        "Could not resolve authentication method. Expected one of api_key, "
        "auth_token, or credentials to be set."
    )
    provider = AnthropicProvider(StubClient(raises=auth_error))

    with pytest.raises(ProviderError) as exc:
        provider.complete(request())

    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "ant auth login" in message
    assert "mock" in message


def test_a_genuine_type_error_is_not_mistaken_for_missing_credentials():
    provider = AnthropicProvider(StubClient(raises=TypeError("bad argument 'foo'")))
    with pytest.raises(ProviderError) as exc:
        provider.complete(request())
    assert "bad argument" in str(exc.value)
    assert "ANTHROPIC_API_KEY" not in str(exc.value)


def test_latency_is_measured():
    client = StubClient(Message(content=[Block(type="text", text="x")]))
    response = AnthropicProvider(client).complete(request())
    assert response.latency_ms >= 0.0
