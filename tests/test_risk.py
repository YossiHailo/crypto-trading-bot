"""Position sizing, reward:risk, and the entry gates."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from agents.risk import RiskManager
from core.models import RiskAction, SignalAction
from tests.conftest import long_position, make_settings, portfolio, sentiment, technical

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _manager(**overrides: object) -> RiskManager:
    defaults = dict(
        paper_trading=True,
        fee_rate=0.0,
        cash_utilization=1.0,
        min_notional=0.0,
        max_notional_pct=1.0,
        risk_per_trade=0.02,
        reward_risk_ratio=2.0,
        min_reward_risk=1.5,
        min_confidence=0.55,
        atr_multiplier=1.5,
        use_atr_stops=True,
        fixed_stop_pct=0.02,
        max_open_positions=3,
        daily_max_loss_pct=0.05,
        cooldown_minutes=60,
        allow_short=False,
        slippage_pct=0.0,
        max_total_exposure_pct=1.0,
    )
    defaults.update(overrides)
    return RiskManager(make_settings(**defaults))


def test_atr_size_stop_and_target() -> None:
    decision = _manager().decide(technical(), sentiment(), portfolio(), now=NOW)
    assert decision.action == RiskAction.EXECUTE_BUY
    expected_size = math.floor((200.0 / 3.0) * 1e8) / 1e8
    assert decision.position_size == expected_size
    assert decision.entry == 100
    assert decision.stop_loss == 97
    assert decision.take_profit == 106
    assert decision.reward_risk == 2
    assert decision.risk_amount == expected_size * 3
    assert "ATR stop" in decision.reasoning


def test_fixed_percent_stop_and_atr_fallback() -> None:
    fixed = _manager(use_atr_stops=False).decide(technical(), sentiment(), portfolio(), now=NOW)
    assert fixed.position_size == 100
    assert fixed.stop_loss == 98
    assert fixed.take_profit == 104
    fallback = _manager().decide(technical(atr=0.0), sentiment(), portfolio(), now=NOW)
    assert fallback.action == RiskAction.EXECUTE_BUY
    assert "fixed" in fallback.reasoning
    assert fallback.stop_loss == 98


def test_cash_cap_shrinks_a_tight_stop() -> None:
    decision = _manager(atr_multiplier=1.0).decide(
        technical(atr=0.01),
        sentiment(),
        portfolio(),
        now=NOW,
    )
    assert decision.action == RiskAction.EXECUTE_BUY
    assert decision.position_size == 100
    assert "capped" in decision.reasoning
    assert decision.risk_amount < 200


def test_low_confidence_and_reward_ratio_block_entries() -> None:
    low = _manager().decide(technical(confidence=0.4), sentiment(), portfolio(), now=NOW)
    assert low.action == RiskAction.NO_ACTION
    assert "minimum" in low.reasoning
    poor = _manager(reward_risk_ratio=1.0, min_reward_risk=1.5).decide(
        technical(), sentiment(), portfolio(), now=NOW
    )
    assert poor.action == RiskAction.NO_ACTION
    assert "Reward:risk" in poor.reasoning


def test_circuit_breaker_blocks_entries_but_not_exits() -> None:
    account = portfolio(equity=9_400)
    blocked = _manager().decide(technical(), sentiment(), account, now=NOW)
    assert blocked.action == RiskAction.NO_ACTION
    assert "circuit breaker" in blocked.reasoning
    allowed = _manager().decide(technical(), sentiment(), portfolio(equity=9_600), now=NOW)
    assert allowed.action == RiskAction.EXECUTE_BUY
    closing = _manager().decide(
        technical(action=SignalAction.SELL, confidence=0.1),
        sentiment(score=1.0),
        portfolio(equity=9_000, positions={"BTC/USDT": long_position()}),
        now=NOW,
    )
    assert closing.action == RiskAction.EXECUTE_SELL
    assert closing.position_size == 1
    assert "Closing" in closing.reasoning


def test_cooldown_and_capacity_gates() -> None:
    cooling = portfolio(cooldowns={"BTC/USDT": NOW + timedelta(minutes=30)})
    blocked = _manager().decide(technical(), sentiment(), cooling, now=NOW)
    assert blocked.action == RiskAction.NO_ACTION
    assert "Cooldown" in blocked.reasoning
    expired = portfolio(cooldowns={"BTC/USDT": NOW - timedelta(minutes=1)})
    assert _manager().decide(technical(), sentiment(), expired, now=NOW).action == RiskAction.EXECUTE_BUY

    positions = {f"ETH{index}/USDT": long_position(symbol=f"ETH{index}/USDT") for index in range(3)}
    full = _manager().decide(technical(), sentiment(), portfolio(positions=positions), now=NOW)
    assert full.action == RiskAction.NO_ACTION
    assert "Max open" in full.reasoning
    pending = portfolio(pending_symbols={"SOL/USDT", "XRP/USDT", "DOGE/USDT"})
    assert _manager().decide(technical(), sentiment(), pending, now=NOW).action == RiskAction.NO_ACTION


def test_sentiment_veto_and_agreement_boost() -> None:
    veto = _manager().decide(technical(), sentiment(score=-0.8), portfolio(), now=NOW)
    assert veto.action == RiskAction.NO_ACTION
    assert "veto" in veto.reasoning
    boosted = _manager(min_confidence=0.55).decide(
        technical(confidence=0.50),
        sentiment(score=1.0),
        portfolio(),
        now=NOW,
    )
    assert boosted.action == RiskAction.EXECUTE_BUY
    assert boosted.confidence == pytest.approx(0.575)


def test_short_rules() -> None:
    flat_sell = _manager().decide(technical(action=SignalAction.SELL), sentiment(score=-1), portfolio(), now=NOW)
    assert flat_sell.action == RiskAction.NO_ACTION
    assert "shorting is disabled" in flat_sell.reasoning
    opened = _manager(allow_short=True).decide(
        technical(action=SignalAction.SELL),
        sentiment(score=-1),
        portfolio(),
        now=NOW,
    )
    assert opened.action == RiskAction.EXECUTE_SELL
    assert opened.stop_loss == 103
    assert opened.take_profit == 94
    live_spot = _manager(
        allow_short=True,
        paper_trading=False,
        live_trading_confirm=True,
        exchange_api_key="k",
        exchange_api_secret="s",
        market_type="spot",
    ).decide(technical(action=SignalAction.SELL), sentiment(score=-1), portfolio(), now=NOW)
    assert live_spot.action == RiskAction.NO_ACTION
    assert "swap" in live_spot.reasoning


def test_hold_and_duplicate_position() -> None:
    hold = _manager().decide(technical(action=SignalAction.HOLD, confidence=0.0), sentiment(), portfolio(), now=NOW)
    assert hold.action == RiskAction.NO_ACTION
    assert "HOLD" in hold.reasoning
    duplicate = _manager().decide(
        technical(),
        sentiment(),
        portfolio(positions={"BTC/USDT": long_position()}),
        now=NOW,
    )
    assert duplicate.action == RiskAction.NO_ACTION
    assert "already open" in duplicate.reasoning


def test_unsynced_live_balance_blocks_entries() -> None:
    decision = _manager().decide(
        technical(),
        sentiment(),
        portfolio(balance_synced=False),
        now=NOW,
    )
    assert decision.action == RiskAction.NO_ACTION
    assert "synced" in decision.reasoning

