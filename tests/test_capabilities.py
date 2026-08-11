from __future__ import annotations

from datetime import date, timedelta

import pytest

from regressa import capabilities


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(capabilities.UnknownModelError):
        capabilities.get("gpt-4-turbo")


def test_mock_models_resolve_to_a_zero_cost_entry():
    caps = capabilities.get("mock-anything")
    assert caps.input_per_mtok == 0.0
    assert caps.supports_sampling is True


def test_price_uses_per_mtok_rates():
    # 1M in + 1M out on Opus 5 = $5 + $25.
    assert capabilities.price("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_cache_reads_are_a_tenth_of_input():
    cost = capabilities.price("claude-opus-5", 0, 0, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Promotional pricing
#
# An intro rate written straight into the headline price is wrong the day the
# promotion ends, and wrong silently — every cost figure and `cost` assertion
# drifts with no code change to blame. These pin the date boundary.
# --------------------------------------------------------------------------- #

PROMO_END = date(2026, 8, 31)


def test_promotional_rate_applies_during_the_window():
    # Sonnet 5 intro pricing is $2/$10 rather than the $3/$15 list rate.
    cost = capabilities.price(
        "claude-sonnet-5", 1_000_000, 1_000_000, on=PROMO_END - timedelta(days=20)
    )
    assert cost == pytest.approx(12.0)


def test_promotional_rate_applies_on_its_final_day_inclusive():
    rates = capabilities.get("claude-sonnet-5").rates(PROMO_END)
    assert rates.promo_active is True
    assert rates.input_per_mtok == pytest.approx(2.00)


def test_list_price_resumes_the_day_after_the_promotion_ends():
    day_after = PROMO_END + timedelta(days=1)
    rates = capabilities.get("claude-sonnet-5").rates(day_after)
    assert rates.promo_active is False
    assert rates.input_per_mtok == pytest.approx(3.00)
    assert capabilities.price(
        "claude-sonnet-5", 1_000_000, 1_000_000, on=day_after
    ) == pytest.approx(18.0)


def test_cache_rates_follow_the_promotional_input_rate():
    # A cache read is 0.1x whichever input rate is actually in force.
    during = capabilities.get("claude-sonnet-5").rates(PROMO_END)
    after = capabilities.get("claude-sonnet-5").rates(PROMO_END + timedelta(days=1))
    assert during.cache_read_per_mtok == pytest.approx(0.20)
    assert after.cache_read_per_mtok == pytest.approx(0.30)


def test_models_without_a_promotion_are_unaffected_by_date():
    a = capabilities.price("claude-opus-5", 1_000_000, 0, on=date(2026, 1, 1))
    b = capabilities.price("claude-opus-5", 1_000_000, 0, on=date(2027, 1, 1))
    assert a == b == pytest.approx(5.0)
    assert capabilities.get("claude-opus-5").rates().promo_active is False


def test_active_promos_reports_the_window_and_goes_quiet_after_it():
    during = dict(capabilities.active_promos(PROMO_END))
    assert "claude-sonnet-5" in during
    assert during["claude-sonnet-5"].promo_until == PROMO_END

    after = capabilities.active_promos(PROMO_END + timedelta(days=1))
    assert after == []


def test_list_price_is_still_reachable_while_a_promo_is_active():
    # Reports show promo/list side by side, so the list rate must not be lost.
    caps = capabilities.get("claude-sonnet-5")
    assert caps.input_per_mtok == pytest.approx(3.00)
    assert caps.rates(PROMO_END).input_per_mtok == pytest.approx(2.00)


def test_temperature_is_rejected_on_models_that_400_on_it():
    issues = capabilities.validate_target("t", "claude-opus-5", temperature=0.0)
    assert len(issues) == 1
    assert "temperature" in issues[0].message


def test_temperature_is_allowed_on_models_that_accept_it():
    assert capabilities.validate_target("t", "claude-sonnet-4-6", temperature=0.2) == []


def test_disabled_thinking_above_high_effort_is_rejected_on_opus_5():
    issues = capabilities.validate_target(
        "t", "claude-opus-5", thinking="disabled", effort="xhigh"
    )
    assert len(issues) == 1
    assert "effort" in issues[0].message


def test_disabled_thinking_at_high_effort_is_fine_on_opus_5():
    assert (
        capabilities.validate_target("t", "claude-opus-5", thinking="disabled", effort="high")
        == []
    )


def test_fable_rejects_any_explicit_thinking_config():
    issues = capabilities.validate_target("t", "claude-fable-5", thinking="disabled")
    assert len(issues) == 1
    assert "always on" in issues[0].message


def test_effort_is_rejected_on_models_without_the_parameter():
    issues = capabilities.validate_target("t", "claude-haiku-4-5", effort="high")
    assert len(issues) == 1
    assert "does not support `effort`" in issues[0].message


def test_max_tokens_ceiling_is_enforced():
    issues = capabilities.validate_target("t", "claude-haiku-4-5", max_tokens=100_000)
    assert len(issues) == 1
    assert "ceiling" in issues[0].message
