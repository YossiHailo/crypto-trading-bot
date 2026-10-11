"""Paper fills, stops, targets, limits, and SQLite reload."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agents.execution import ExecutionAgent, PaperExchange
from agents.risk import RiskManager
from core.models import CloseReason, RiskAction
from core.state import PortfolioStore
from tests.conftest import make_settings, sentiment, technical

NOW = datetime(2026, 10, 6, 15, tzinfo=timezone.utc)


def _settings(tmp_path, **overrides):
    defaults = dict(
        paper_trading=True,
        fee_rate=0.001,
        order_type="market",
        cooldown_minutes=60,
        data_dir=str(tmp_path),
        starting_balance=10_000,
        cash_utilization=1,
        min_notional=0,
        min_confidence=0.55,
        risk_per_trade=0.02,
        atr_multiplier=1.5,
        use_atr_stops=True,
        # Exact accounting below assumes no slippage and the original caps.
        slippage_pct=0.0,
        max_notional_pct=1.0,
        reward_risk_ratio=2.0,
    )
    defaults.update(overrides)
    return make_settings(**defaults)


def _store(tmp_path, settings, name="paper.sqlite"):
    return PortfolioStore(
        tmp_path / name,
        settings.starting_balance,
        fee_rate=settings.fee_rate,
        clock=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_market_buy_then_stop_and_target(tmp_path) -> None:
    settings = _settings(tmp_path, fee_rate=0.001)
    store = _store(tmp_path, settings)
    paper = PaperExchange(settings, store)
    buy = technical()
    # Size 1 is passed directly so the fee math stays obvious.
    from core.models import RiskDecision

    decision = RiskDecision(
        action=RiskAction.EXECUTE_BUY,
        symbol="BTC/USDT",
        position_size=1,
        entry=100,
        stop_loss=95,
        take_profit=110,
        confidence=0.9,
        reasoning="sized by the test",
    )
    filled = await paper.execute(decision, market_price=100)
    assert filled.success and filled.opened
    assert store.view().cash == pytest.approx(9_899.9)
    assert "BTC/USDT" in store.view().positions

    stopped = await paper.mark_prices({"BTC/USDT": 90})
    assert len(stopped.closed) == 1
    trade = stopped.closed[0]
    assert trade.reason == CloseReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(90)
    assert trade.pnl == pytest.approx(-10.19)
    assert store.view().cash == pytest.approx(9_989.81)
    assert store.view({"BTC/USDT": 90}).cooldowns["BTC/USDT"] > NOW
    await paper.close()
    store.close()


@pytest.mark.asyncio
async def test_take_profit_fills_at_the_target_without_cooldown(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = _store(tmp_path, settings, "tp.sqlite")
    paper = PaperExchange(settings, store)
    from core.models import RiskDecision

    await paper.execute(
        RiskDecision(
            action=RiskAction.EXECUTE_BUY,
            symbol="BTC/USDT",
            position_size=1,
            entry=100,
            stop_loss=95,
            take_profit=110,
            confidence=0.9,
            reasoning="target",
        ),
        market_price=100,
    )
    marked = await paper.mark_prices({"BTC/USDT": 115})
    trade = marked.closed[0]
    assert trade.reason == CloseReason.TAKE_PROFIT
    assert trade.exit_price == pytest.approx(110)
    assert trade.pnl == pytest.approx(9.79)
    assert store.view().cooldowns == {}
    store.close()


@pytest.mark.asyncio
async def test_insufficient_funds_and_reload(tmp_path) -> None:
    settings = _settings(tmp_path, fee_rate=0.0)
    path = tmp_path / "persist.sqlite"
    store = PortfolioStore(path, 10_000, fee_rate=0.0, clock=lambda: NOW)
    paper = PaperExchange(settings, store)
    from core.models import RiskDecision

    rejected = await paper.execute(
        RiskDecision(
            action=RiskAction.EXECUTE_BUY,
            symbol="BTC/USDT",
            position_size=1_000,
            entry=100,
            stop_loss=95,
            take_profit=110,
            confidence=0.9,
            reasoning="too big",
        )
    )
    assert rejected.success is False
    assert rejected.status == "insufficient_funds"
    assert store.view().cash == pytest.approx(10_000)

    opened = await paper.execute(
        RiskDecision(
            action=RiskAction.EXECUTE_BUY,
            symbol="ETH/USDT",
            position_size=2,
            entry=50,
            stop_loss=45,
            take_profit=60,
            confidence=0.9,
            reasoning="small",
        )
    )
    assert opened.opened
    cash = store.view().cash
    store.close()

    reloaded = PortfolioStore(path, 1, fee_rate=0.0, clock=lambda: NOW)
    assert reloaded.view().cash == pytest.approx(cash)
    assert reloaded.get_position("ETH/USDT").size == pytest.approx(2)
    reloaded.close()


@pytest.mark.asyncio
async def test_limit_rests_until_price_touches(tmp_path) -> None:
    settings = _settings(tmp_path, order_type="limit", fee_rate=0.0)
    store = _store(tmp_path, settings, "limit.sqlite")
    paper = PaperExchange(settings, store)
    from core.models import RiskDecision

    resting = await paper.execute(
        RiskDecision(
            action=RiskAction.EXECUTE_BUY,
            symbol="BTC/USDT",
            position_size=1,
            entry=100,
            stop_loss=95,
            take_profit=110,
            confidence=0.9,
            reasoning="limit",
        ),
        market_price=105,
    )
    assert resting.status == "open"
    assert store.get_position("BTC/USDT") is None
    assert store.view().cash == pytest.approx(10_000)

    still = await paper.mark_prices({"BTC/USDT": 101})
    assert still.opened == []
    filled = await paper.mark_prices({"BTC/USDT": 100})
    assert filled.opened[0].opened
    assert store.get_position("BTC/USDT").entry == pytest.approx(100)
    store.close()


@pytest.mark.asyncio
async def test_stop_out_arms_cooldown_inside_the_risk_agent(tmp_path) -> None:
    settings = _settings(
        tmp_path,
        fee_rate=0.0,
        cooldown_minutes=60,
        min_confidence=0.5,
    )
    store = _store(tmp_path, settings, "cool.sqlite")
    agent = ExecutionAgent(settings, store)
    risk = RiskManager(settings)
    from core.models import RiskDecision

    signal = technical(confidence=0.9, atr=2, price=100)
    first = risk.decide(signal, sentiment(), store.view(), now=NOW)
    assert first.action == RiskAction.EXECUTE_BUY
    await agent.execute(first, market_price=100)
    # Fill at the stop. A gap to 90 would also trip the daily loss limit.
    await agent.mark_prices({"BTC/USDT": 97})
    blocked = risk.decide(signal, sentiment(), store.view({"BTC/USDT": 90}), now=NOW)
    assert blocked.action == RiskAction.NO_ACTION
    assert "Cooldown" in blocked.reasoning
    later = risk.decide(
        signal,
        sentiment(),
        store.view(),
        now=NOW + timedelta(minutes=61),
    )
    assert later.action == RiskAction.EXECUTE_BUY
    await agent.close()


@pytest.mark.asyncio
async def test_day_roll_resets_realized_pnl(tmp_path) -> None:
    clock = {"now": NOW}
    settings = _settings(tmp_path, fee_rate=0.0)
    store = PortfolioStore(
        tmp_path / "day.sqlite",
        10_000,
        fee_rate=0.0,
        clock=lambda: clock["now"],
    )
    paper = PaperExchange(settings, store)
    from core.models import RiskDecision

    await paper.execute(
        RiskDecision(
            action=RiskAction.EXECUTE_BUY,
            symbol="BTC/USDT",
            position_size=1,
            entry=100,
            stop_loss=95,
            take_profit=110,
            confidence=1,
            reasoning="day",
        )
    )
    await paper.mark_prices({"BTC/USDT": 110})
    assert store.view().realized_pnl_today > 0
    clock["now"] = NOW + timedelta(days=1)
    summary = store.roll_day_if_needed({})
    assert summary is not None
    assert "2026-10-06" in summary
    view = store.view()
    assert view.day_start_date == "2026-10-07"
    assert view.realized_pnl_today == 0
    store.close()


@pytest.mark.asyncio
async def test_short_stop(tmp_path) -> None:
    settings = _settings(tmp_path, fee_rate=0.0, allow_short=True)
    store = _store(tmp_path, settings, "short.sqlite")
    paper = PaperExchange(settings, store)
    from core.models import RiskDecision

    opened = await paper.execute(
        RiskDecision(
            action=RiskAction.EXECUTE_SELL,
            symbol="BTC/USDT",
            position_size=1,
            entry=100,
            stop_loss=103,
            take_profit=94,
            confidence=0.9,
            reasoning="short",
        )
    )
    assert opened.opened
    assert store.get_position("BTC/USDT").side == "short"
    marked = await paper.mark_prices({"BTC/USDT": 110})
    assert marked.closed[0].reason == CloseReason.STOP_LOSS
    assert marked.closed[0].pnl == pytest.approx(-10)
    assert store.view().cash == pytest.approx(9_990)
    store.close()
