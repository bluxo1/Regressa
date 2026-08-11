"""The live-integration checker, exercised with stub clients.

These tests verify the checker's own logic — that it reports the right verdict for
a given API behaviour. They cannot verify the API itself; that is the whole point
of `regressa doctor` and requires a key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from regressa import doctor
from regressa.assertions import judge
from regressa.providers.anthropic_provider import AnthropicProvider
from regressa.providers.base import ProviderError


@dataclass
class Block:
    type: str
    text: str = ""


@dataclass
class Usage:
    input_tokens: int = 12
    output_tokens: int = 4
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Message:
    content: list[Block] = field(default_factory=lambda: [Block("text", "OK")])
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = "end_turn"
    stop_details: Any = None


class StubMessages:
    def __init__(self, message: Any, raises: Exception | None = None) -> None:
        self._message = message
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if self._raises:
            raise self._raises
        return self._message

    def stream(self, **params: Any):
        self.calls.append(params)
        message = self._message

        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return None

            def get_final_message(self):
                return message

        return Ctx()


class StubClient:
    def __init__(self, message: Any = None, raises: Exception | None = None) -> None:
        self.messages = StubMessages(message or Message(), raises)


@pytest.fixture(autouse=True)
def _reset_judge_client():
    yield
    judge.set_client(None)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def test_basic_request_check_passes_on_a_well_formed_response():
    provider = AnthropicProvider(StubClient())
    check, response = doctor.check_basic_request(provider, "claude-haiku-4-5")
    assert check.passed is True
    assert check.cost_usd > 0
    assert response is not None


def test_basic_request_check_fails_when_usage_is_missing():
    # A response that parses but carries no usage would silently zero every cost.
    message = Message(usage=Usage(input_tokens=0, output_tokens=0))
    check, _ = doctor.check_basic_request(
        AnthropicProvider(StubClient(message)), "claude-haiku-4-5"
    )
    assert check.passed is False
    assert "input_tokens" in check.detail


def test_basic_request_check_fails_on_empty_text():
    message = Message(content=[])
    check, _ = doctor.check_basic_request(
        AnthropicProvider(StubClient(message)), "claude-haiku-4-5"
    )
    assert check.passed is False
    assert "empty" in check.detail


def test_a_provider_error_is_reported_not_raised():
    provider = AnthropicProvider(StubClient(raises=RuntimeError("boom")))
    check, response = doctor.check_basic_request(provider, "claude-haiku-4-5")
    assert check.passed is False
    assert "boom" in check.detail
    assert response is None


def test_effort_check_is_skipped_for_models_without_the_parameter():
    check = doctor.check_effort(AnthropicProvider(StubClient()), "claude-haiku-4-5")
    assert check.skipped is True
    assert "no effort parameter" in check.detail


def test_effort_check_runs_for_models_that_support_it():
    client = StubClient()
    check = doctor.check_effort(AnthropicProvider(client), "claude-sonnet-5")
    assert check.passed is True
    assert client.messages.calls[0]["output_config"] == {"effort": "low"}


def test_streaming_check_takes_the_streaming_branch():
    client = StubClient()
    check = doctor.check_streaming(AnthropicProvider(client), "claude-haiku-4-5")
    assert check.passed is True
    # Above the threshold the provider must stream, not create.
    assert client.messages.calls[0]["max_tokens"] > 16_000


def test_structured_output_check_passes_on_valid_grader_json():
    import json

    judge.set_client(
        StubClient(Message(content=[Block("text", json.dumps({"score": 1.0, "reasoning": "ok"}))]))
    )
    check = doctor.check_structured_output("claude-haiku-4-5")
    assert check.passed is True


def test_structured_output_check_fails_when_the_schema_is_not_honoured():
    import json

    judge.set_client(StubClient(Message(content=[Block("text", json.dumps({"score": 1.0}))])))
    check = doctor.check_structured_output("claude-haiku-4-5")
    assert check.passed is False
    assert "missing required keys" in check.detail


def test_structured_output_check_fails_on_unparseable_output():
    judge.set_client(StubClient(Message(content=[Block("text", "looks good to me")])))
    check = doctor.check_structured_output("claude-haiku-4-5")
    assert check.passed is False
    assert "not parseable" in check.detail


# --------------------------------------------------------------------------- #
# The capability-table check — the most consequential one
# --------------------------------------------------------------------------- #


def test_sampling_check_is_skipped_for_models_that_accept_sampling():
    check = doctor.check_sampling_is_rejected("claude-sonnet-4-6")
    assert check.skipped is True


def test_sampling_check_passes_when_the_api_rejects_temperature(monkeypatch):
    import anthropic

    class Rejecting:
        class messages:  # noqa: N801
            @staticmethod
            def create(**_params):
                raise anthropic.BadRequestError(
                    "temperature not supported",
                    response=_FakeResponse(400),
                    body=None,
                )

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: Rejecting())
    check = doctor.check_sampling_is_rejected("claude-opus-5")
    assert check.passed is True
    assert "rejected temperature as expected" in check.detail


def test_sampling_check_fails_loudly_if_the_api_accepts_temperature(monkeypatch):
    """If this ever fires, capabilities.py is wrong and blocking valid targets."""
    import anthropic

    class Accepting:
        class messages:  # noqa: N801
            @staticmethod
            def create(**_params):
                return Message()

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: Accepting())
    check = doctor.check_sampling_is_rejected("claude-opus-5")
    assert check.passed is False
    assert "ACCEPTED temperature" in check.detail
    assert "supports_sampling=True" in check.detail


class _FakeResponse:
    """Minimal stand-in for an httpx response, for constructing SDK errors."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.request = None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_the_default_model_exercises_the_checks_that_matter():
    """Guards against reverting to a cheap model that skips the useful checks.

    Haiku 4.5 accepts sampling parameters and has no effort parameter, so using it
    as the default silently skipped both the capability-table check and the effort
    check — the two things `doctor` exists to verify.
    """
    from regressa import capabilities

    caps = capabilities.get(doctor.DEFAULT_MODEL)
    assert caps.supports_sampling is False, "default must exercise the sampling check"
    assert caps.effort_levels, "default must exercise the effort check"


def test_low_credit_is_reported_as_an_account_problem_not_a_bad_request(monkeypatch):
    """A 400 for an empty balance must not read as a malformed request."""
    import anthropic

    from regressa.providers.anthropic_provider import LOW_CREDIT_HINT

    billing_error = anthropic.BadRequestError(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Your credit balance is too low to access the Anthropic API.'}}",
        response=_FakeResponse(400),
        body=None,
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: StubClient())

    provider = AnthropicProvider(StubClient(raises=billing_error))
    diag = doctor.diagnose("claude-sonnet-5", provider=provider)

    assert diag.ok is False
    names = [c.name for c in diag.checks]
    # One clear account-balance failure, not the same 400 repeated per check.
    assert names == ["anthropic sdk", "account balance"]
    assert diag.checks[-1].detail == LOW_CREDIT_HINT
    assert "Plans & Billing" in diag.checks[-1].detail
    assert diag.cost_usd == 0.0


def test_low_credit_surfaces_through_a_normal_run_too(store):
    """Not just doctor: a suite run must explain it rather than dumping JSON."""
    import anthropic

    from regressa.config import Suite
    from regressa.runner import Runner

    billing_error = anthropic.BadRequestError(
        "Error code: 400 - {'message': 'Your credit balance is too low to access "
        "the Anthropic API.'}",
        response=_FakeResponse(400),
        body=None,
    )

    class Billing:
        name = "anthropic"

        def complete(self, request):
            return AnthropicProvider(StubClient(raises=billing_error)).complete(request)

    suite = Suite.model_validate(
        {
            "name": "billing",
            "targets": [{"id": "t", "provider": "anthropic", "model": "claude-sonnet-5"}],
            "cases": [{"id": "c", "prompt": "hi"}],
        }
    )
    runner = Runner(store)
    runner._providers["anthropic"] = Billing()
    run = runner.run(suite)

    assert run.errors == 1
    assert "credit balance" in run.results[0].error
    assert "Plans & Billing" in run.results[0].error


def test_diagnose_short_circuits_on_missing_credentials():
    """One clear credentials failure beats six copies of the same error.

    The SDK resolves auth lazily, so a client constructs fine without a key and
    the failure only appears on the first request.
    """
    auth_error = TypeError(
        "Could not resolve authentication method. Expected one of api_key, "
        "auth_token, or credentials to be set."
    )
    provider = AnthropicProvider(StubClient(raises=auth_error))

    diag = doctor.diagnose("claude-haiku-4-5", provider=provider)
    assert diag.ok is False
    names = [c.name for c in diag.checks]
    # sdk check passes, then a single credentials failure, then nothing.
    assert names == ["anthropic sdk", "credentials"]
    assert "ANTHROPIC_API_KEY" in diag.checks[-1].detail


def test_diagnose_sums_cost_across_checks(monkeypatch):
    import json

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: StubClient())
    judge.set_client(
        StubClient(Message(content=[Block("text", json.dumps({"score": 1.0, "reasoning": "x"}))]))
    )

    diag = doctor.diagnose("claude-haiku-4-5", provider=AnthropicProvider(StubClient()))
    assert diag.ok is True, [c.detail for c in diag.failures]
    assert diag.cost_usd > 0
    # Haiku has no effort parameter, so that check reports as skipped.
    assert any(c.skipped for c in diag.checks)
