"""The cycle keeps going when one symbol or the sentiment agent fails."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agents.execution import ExecutionAgent
from agents.risk import RiskManager
from core.models import RiskAction, SignalAction
from core.orchestrator import run_cycle
from core.state import PortfolioStore
from notifications.telegram import TelegramNotifier
from tests.conftest import make_settings, sentiment, technical

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


class ScriptedMarket:
    def __init__(self, scripts: dict[str, list[object]]) -> None:
        self.scripts = scripts
        self.calls = {symbol: 0 for symbol in scripts}

    async def analyze(self, symbol: str):
        queue = self.scripts[symbol]
        item = queue[min(self.calls[symbol], len(queue) - 1)]
        self.calls[symbol] += 1
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        return None


class ScriptedSentiment:
    def __init__(self, result: object) -> None:
        self.result = result

    async def evaluate(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def close(self) -> None:
        return None


class RecordingNotifier:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def info(self, text: str) -> bool:
        self.events.append(("info", text))
        return True

    async def error(self, message: str) -> bool:
        self.events.append(("error", message))
        return True

    async def trade_opened(self, decision, result) -> bool:
        self.events.append(("open", decision.symbol))
        return True

    async def trade_closed(self, trade) -> bool:
        self.events.append(("close", f"{trade.symbol}:{trade.reason.value}"))
        return True

    async def daily_summary(self, text: str) -> bool:
        self.events.append(("summary", text))
        return True

    async def close(self) -> None:
        return None


def _settings(tmp_path, **overrides):
    defaults = dict(
        paper_trading=True,
        fee_rate=0.0,
        cash_utilization=1.0,
        min_notional=0.0,
        max_notional_pct=1.0,
        risk_per_trade=0.02,
        min_confidence=0.55,
        atr_multiplier=1.5,
        use_atr_stops=True,
        data_dir=str(tmp_path),
        symbols="BTC/USDT",
        request_timeout_seconds=5,
        slippage_pct=0.0,
        reward_risk_ratio=2.0,
    )
    defaults.update(overrides)
    return make_settings(**defaults)


@pytest.mark.asyncio
async def test_one_symbol_failing_does_not_stop_the_other(tmp_path) -> None:
    settings = _settings(tmp_path, symbols="BAD/USDT,ETH/USDT")
    store = PortfolioStore(tmp_path / "iso.sqlite", 10_000, fee_rate=0.0, clock=lambda: NOW)
    execution = ExecutionAgent(settings, store)
    market = ScriptedMarket(
        {
            "BAD/USDT": [RuntimeError("exchange down")],
            "ETH/USDT": [technical(symbol="ETH/USDT", action=SignalAction.HOLD, confidence=0.0, price=50)],
        }
    )
    report = await run_cycle(
        symbols=["BAD/USDT", "ETH/USDT"],
        market=market,
        sentiment_agent=ScriptedSentiment(sentiment()),
        risk=RiskManager(settings),
        execution=execution,
        notifier=RecordingNotifier(),
        settings=settings,
    )
    assert len(report.errors) == 1
    assert "BAD/USDT" in report.errors[0]
    assert [item.symbol for item in report.decisions] == ["ETH/USDT"]
    assert report.decisions[0].action == RiskAction.NO_ACTION
    await execution.close()


@pytest.mark.asyncio
async def test_sentiment_crash_degrades_and_a_buy_still_fills(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = PortfolioStore(tmp_path / "buy.sqlite", 10_000, fee_rate=0.0, clock=lambda: NOW)
    execution = ExecutionAgent(settings, store)
    notifier = RecordingNotifier()
    market = ScriptedMarket({"BTC/USDT": [technical()]})
    report = await run_cycle(
        symbols=["BTC/USDT"],
        market=market,
        sentiment_agent=ScriptedSentiment(RuntimeError("sentiment boom")),
        risk=RiskManager(settings),
        execution=execution,
        notifier=notifier,
        settings=settings,
    )
    assert report.sentiment is not None
    assert report.sentiment.score == 0
    assert report.decisions[0].action == RiskAction.EXECUTE_BUY
    assert store.get_position("BTC/USDT") is not None
    assert ("open", "BTC/USDT") in notifier.events
    await execution.close()


@pytest.mark.asyncio
async def test_take_profit_notifies_on_the_next_cycle(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = PortfolioStore(tmp_path / "tp.sqlite", 10_000, fee_rate=0.0, clock=lambda: NOW)
    execution = ExecutionAgent(settings, store)
    notifier = RecordingNotifier()
    buy = technical(price=100, atr=2, confidence=0.9)
    hold = technical(action=SignalAction.HOLD, confidence=0.0, price=110, atr=2)
    market = ScriptedMarket({"BTC/USDT": [buy, hold]})
    first = await run_cycle(
        symbols=["BTC/USDT"],
        market=market,
        sentiment_agent=ScriptedSentiment(sentiment()),
        risk=RiskManager(settings),
        execution=execution,
        notifier=notifier,
        settings=settings,
    )
    assert first.decisions[0].action == RiskAction.EXECUTE_BUY
    second = await run_cycle(
        symbols=["BTC/USDT"],
        market=market,
        sentiment_agent=ScriptedSentiment(sentiment()),
        risk=RiskManager(settings),
        execution=execution,
        notifier=notifier,
        settings=settings,
    )
    assert second.closed_trades
    assert second.closed_trades[0].reason.value == "take_profit"
    assert any(event[0] == "close" for event in notifier.events)
    assert store.get_position("BTC/USDT") is None
    await execution.close()


@pytest.mark.asyncio
async def test_daily_summary_is_emitted_when_the_utc_day_changes(tmp_path) -> None:
    clock = {"now": NOW}
    settings = _settings(tmp_path)
    store = PortfolioStore(tmp_path / "sum.sqlite", 10_000, fee_rate=0.0, clock=lambda: clock["now"])
    execution = ExecutionAgent(settings, store)
    notifier = RecordingNotifier()
    market = ScriptedMarket(
        {"BTC/USDT": [technical(action=SignalAction.HOLD, confidence=0.0, price=100)]}
    )

    async def cycle():
        return await run_cycle(
            symbols=["BTC/USDT"],
            market=market,
            sentiment_agent=ScriptedSentiment(sentiment()),
            risk=RiskManager(settings),
            execution=execution,
            notifier=notifier,
            settings=settings,
        )

    first = await cycle()
    assert first.daily_summary is None
    clock["now"] = NOW + timedelta(days=1)
    second = await cycle()
    assert second.daily_summary is not None
    assert "2026-10-06" in second.daily_summary
    assert any(event[0] == "summary" for event in notifier.events)
    await execution.close()


@pytest.mark.asyncio
async def test_telegram_is_a_noop_without_credentials() -> None:
    async def transport(_text: str) -> None:
        raise AssertionError("disabled notifier must not send")

    notifier = TelegramNotifier("", "", transport=transport)
    assert notifier.enabled is False
    assert await notifier.info("hello") is False
    assert await notifier.error("boom") is False
    assert await notifier.daily_summary("day") is False
