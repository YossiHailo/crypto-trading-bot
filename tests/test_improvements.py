"""Tests for the v1.1 fixes: closed candles, strategy, caps, paper realism, kill switch, redaction."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from agents.execution import PaperExchange
from agents.market_data import MarketDataAgent
from agents.risk import RiskManager
from agents.sentiment import score_headlines
from agents.strategies import TrendBreakoutStrategy
from core.models import Bar, CloseReason, RiskAction, RiskDecision, SignalAction
from core.redact import RedactingFilter, redact, register_secret
from core.state import PortfolioStore
from core.timeframes import split_closed, timeframe_to_ms
from tests.conftest import long_position, make_settings, portfolio, sentiment, technical

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
HOUR = 3_600_000


# ------------------------------------------------------------ closed candles


def test_timeframe_parsing() -> None:
    assert timeframe_to_ms("15m") == 900_000
    assert timeframe_to_ms("4h") == 4 * HOUR
    with pytest.raises(ValueError):
        timeframe_to_ms("1M")


def test_split_closed_drops_only_the_forming_candle() -> None:
    now_ms = int(NOW.timestamp() * 1000)
    rows = [[now_ms - 3 * HOUR, 1, 1, 1, 1, 1], [now_ms - 2 * HOUR, 1, 1, 1, 1, 1], [now_ms - HOUR // 2, 1, 1, 1, 2, 1]]
    closed, forming = split_closed(rows, "1h", now_ms)
    assert len(closed) == 2 and forming is rows[-1]
    closed, forming = split_closed(rows[:2], "1h", now_ms)
    assert len(closed) == 2 and forming is None


class _Source:
    def __init__(self, rows):
        self.rows = rows
        self.limit = None

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        self.limit = limit
        return self.rows

    async def close(self):
        return None


def _trend_rows(n: int, *, start_ms: int, step: int = HOUR, slope: float = 0.004, jump: float = 0.0):
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(slope + rng.normal(0, 0.002, n)))
    closes[-1] *= 1 + jump
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        rows.append([start_ms + i * step, o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1.0])
    return rows


@pytest.mark.asyncio
async def test_market_agent_scores_closed_bars_but_reports_live_price() -> None:
    now_ms = int(NOW.timestamp() * 1000)
    rows = _trend_rows(260, start_ms=now_ms - 259 * HOUR - HOUR // 2)
    rows[-1][4] = 999.0  # the forming candle has a crazy price
    source = _Source(rows)
    agent = MarketDataAgent(source, "1h", 258, strategy=TrendBreakoutStrategy(), clock=lambda: NOW)
    signal = await agent.analyze("BTC/USDT")
    assert source.limit == 259  # one extra row for the dropped candle
    assert signal.price == 999.0
    assert signal.indicators["close"] == pytest.approx(rows[-2][4])  # scored on the last closed bar
    assert signal.bar_time_ms == rows[-2][0]
    assert signal.recent_bars and signal.recent_bars[-1].open_ms == rows[-1][0]


# ------------------------------------------------------------------ strategy


def _frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_trend_strategy_buys_a_breakout_in_an_uptrend() -> None:
    result = TrendBreakoutStrategy().evaluate(_frame(_trend_rows(260, start_ms=0, jump=0.03)))
    assert result is not None
    assert result.action == SignalAction.BUY
    assert 0.6 <= result.confidence <= 1.0
    assert "breakout" in result.reasoning


def test_trend_strategy_exits_longs_without_signalling_a_short() -> None:
    rows = _trend_rows(260, start_ms=0, jump=-0.08)
    result = TrendBreakoutStrategy().evaluate(_frame(rows))
    assert result.action == SignalAction.HOLD  # still above the 200 EMA, so no short entry
    assert result.exit_long is True


def test_trend_strategy_needs_history() -> None:
    assert TrendBreakoutStrategy().evaluate(_frame(_trend_rows(100, start_ms=0))) is None


# --------------------------------------------------------------------- risk


def _risk(**overrides):
    defaults = dict(fee_rate=0.0, slippage_pct=0.0, min_notional=0.0, cash_utilization=1.0, reward_risk_ratio=2.0)
    defaults.update(overrides)
    return RiskManager(make_settings(**defaults))


def test_default_per_position_cap_prevents_all_in_trades() -> None:
    # A tight stop asks for 3.3x equity; the default cap is 50% per position.
    decision = _risk(risk_per_trade=0.02).decide(technical(atr=0.4), sentiment(), portfolio(), now=NOW)
    assert decision.action == RiskAction.EXECUTE_BUY
    assert decision.position_size * decision.entry == pytest.approx(5_000, rel=1e-6)
    assert "50% of equity per position" in decision.reasoning
    assert "of equity" in decision.reasoning and "target 2.00%" in decision.reasoning


def test_total_exposure_cap_counts_open_positions() -> None:
    account = portfolio(open_notional=9_000.0, positions={"ETH/USDT": long_position("ETH/USDT")})
    decision = _risk().decide(technical(), sentiment(), account, now=NOW)
    assert decision.action == RiskAction.EXECUTE_BUY
    assert decision.position_size * decision.entry == pytest.approx(1_000, rel=1e-6)
    full = portfolio(open_notional=10_000.0, positions={"ETH/USDT": long_position("ETH/USDT")})
    assert _risk().decide(technical(), sentiment(), full, now=NOW).action == RiskAction.NO_ACTION


def test_reward_risk_is_checked_after_costs() -> None:
    # Stop 0.5% away, 2R target: 0.1% fee + 0.05% slippage per side turns 2R into ~1.08R.
    decision = _risk(fee_rate=0.001, slippage_pct=0.0005, min_reward_risk=1.5).decide(
        technical(atr=1 / 3), sentiment(), portfolio(), now=NOW
    )
    assert decision.action == RiskAction.NO_ACTION
    assert "after fees and slippage" in decision.reasoning


def test_pause_and_kill_switch_block_entries_but_not_exits() -> None:
    for account in (portfolio(paused=True), portfolio(halted_reason="max drawdown 20%")):
        assert _risk().decide(technical(), sentiment(), account, now=NOW).action == RiskAction.NO_ACTION
    holding = portfolio(halted_reason="x", positions={"BTC/USDT": long_position()})
    exit_signal = technical(action=SignalAction.HOLD, confidence=0.0)
    exit_signal = exit_signal.__class__(**{**exit_signal.__dict__, "exit_long": True})
    assert _risk().decide(exit_signal, sentiment(), holding, now=NOW).action == RiskAction.EXECUTE_SELL


# ----------------------------------------------------------------- settings


def test_settings_reject_dangerous_values() -> None:
    with pytest.raises(ValueError):
        make_settings(risk_per_trade=0.5)
    with pytest.raises(ValueError):
        make_settings(dashboard_host="0.0.0.0", dashboard_token="")
    with pytest.raises(ValueError):
        make_settings(market_type="spot", leverage=3)
    with pytest.raises(ValueError):
        make_settings(strategy="trend", ohlcv_limit=100)
    assert make_settings(dashboard_host="0.0.0.0", dashboard_token="long-random-token").dashboard_token


# ------------------------------------------------------------ paper realism


def _paper(tmp_path, clock=None, **overrides):
    defaults = dict(fee_rate=0.0, slippage_pct=0.001, data_dir=str(tmp_path), cooldown_minutes=60)
    defaults.update(overrides)
    settings = make_settings(**defaults)
    store = PortfolioStore(tmp_path / "p.sqlite", 10_000, fee_rate=0.0, clock=clock or (lambda: NOW))
    return PaperExchange(settings, store), store


def _buy(size=1.0, entry=100.0):
    return RiskDecision(RiskAction.EXECUTE_BUY, "BTC/USDT", size, entry, entry - 5, entry + 10, 0.8, "t")


@pytest.mark.asyncio
async def test_paper_market_fill_pays_slippage_and_keeps_bracket_distance(tmp_path) -> None:
    paper, store = _paper(tmp_path)
    result = await paper.execute(_buy(), market_price=100.0)
    assert result.price == pytest.approx(100.1)
    position = store.get_position("BTC/USDT")
    assert position.stop_loss == pytest.approx(95.1)
    assert position.take_profit == pytest.approx(110.1)


@pytest.mark.asyncio
async def test_paper_sees_a_wick_through_the_stop_between_checks(tmp_path) -> None:
    paper, store = _paper(tmp_path, slippage_pct=0.0)
    await paper.execute(_buy(), market_price=100.0)
    later = int((NOW + timedelta(hours=1)).timestamp() * 1000)
    wick = (Bar(open_ms=later, close_ms=later + HOUR, high=101.0, low=94.0),)
    marked = await paper.mark_prices({"BTC/USDT": 99.0}, {"BTC/USDT": wick})
    assert len(marked.closed) == 1
    assert marked.closed[0].reason == CloseReason.STOP_LOSS
    assert marked.closed[0].exit_price == pytest.approx(95.0)


@pytest.mark.asyncio
async def test_paper_assumes_stop_first_when_a_bar_touches_both(tmp_path) -> None:
    paper, _ = _paper(tmp_path, slippage_pct=0.0)
    await paper.execute(_buy(), market_price=100.0)
    later = int((NOW + timedelta(hours=1)).timestamp() * 1000)
    both = (Bar(open_ms=later, close_ms=later + HOUR, high=111.0, low=94.0),)
    marked = await paper.mark_prices({"BTC/USDT": 100.0}, {"BTC/USDT": both})
    assert marked.closed[0].reason == CloseReason.STOP_LOSS


@pytest.mark.asyncio
async def test_paper_ignores_wicks_from_before_the_entry(tmp_path) -> None:
    paper, store = _paper(tmp_path, slippage_pct=0.0)
    await paper.execute(_buy(), market_price=100.0)
    earlier = int((NOW - timedelta(minutes=30)).timestamp() * 1000)
    old = (Bar(open_ms=earlier, close_ms=earlier + HOUR, high=101.0, low=90.0),)
    marked = await paper.mark_prices({"BTC/USDT": 100.0}, {"BTC/USDT": old})
    assert not marked.closed
    assert store.get_position("BTC/USDT") is not None


@pytest.mark.asyncio
async def test_paper_limit_expires(tmp_path) -> None:
    clock = {"now": NOW}
    paper, store = _paper(tmp_path, clock=lambda: clock["now"], order_type="limit", pending_order_ttl_minutes=60)
    await paper.execute(_buy(entry=95.0), market_price=100.0)
    assert len(store.pending_orders()) == 1
    clock["now"] = NOW + timedelta(minutes=61)
    marked = await paper.mark_prices({"BTC/USDT": 100.0})
    assert store.pending_orders() == []
    assert any("expired" in e for e in marked.errors)


# --------------------------------------------------------- state and halts


def test_kill_switch_trips_on_drawdown_from_peak_and_resets(tmp_path) -> None:
    store = PortfolioStore(tmp_path / "k.sqlite", 10_000, fee_rate=0.0, clock=lambda: NOW, max_drawdown_pct=0.2)
    store.open_position(symbol="BTC/USDT", side="long", size=100, price=100, stop_loss=50, take_profit=200)
    assert store.update_peak_and_check_drawdown({"BTC/USDT": 120}) is None  # new peak 12,000
    reason = store.update_peak_and_check_drawdown({"BTC/USDT": 95})  # 9,500 is -20.8%
    assert reason and "kill switch" in reason
    assert store.view().halted_reason
    store.reset_halt({"BTC/USDT": 95})
    view = store.view({"BTC/USDT": 95})
    assert view.halted_reason is None and view.peak_equity == pytest.approx(9_500)


def test_old_database_is_migrated(tmp_path) -> None:
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE account (id INTEGER PRIMARY KEY CHECK (id = 1), cash REAL NOT NULL, day_start_equity REAL NOT NULL,
            day_start_date TEXT NOT NULL, realized_pnl_today REAL NOT NULL, balance_synced INTEGER NOT NULL);
        INSERT INTO account VALUES (1, 5000, 5000, '2026-10-06', 0, 1);
        CREATE TABLE positions (symbol TEXT PRIMARY KEY, side TEXT NOT NULL, size REAL NOT NULL, entry REAL NOT NULL,
            stop_loss REAL NOT NULL, take_profit REAL NOT NULL, opened_at TEXT NOT NULL, open_fee REAL NOT NULL,
            margin_locked REAL NOT NULL, exchange_protected INTEGER NOT NULL, order_id TEXT);
        """
    )
    conn.commit()
    conn.close()
    store = PortfolioStore(path, 10_000, clock=lambda: NOW)
    view = store.view()
    assert view.cash == 5000 and view.paused is False and view.halted_reason is None
    store.close()


def test_swap_accounting_locks_margin_not_notional(tmp_path) -> None:
    store = PortfolioStore(tmp_path / "s.sqlite", 1_000, fee_rate=0.0, clock=lambda: NOW, market_type="swap", leverage=5)
    store.open_position(symbol="BTC/USDT", side="long", size=1, price=2_000, stop_loss=1_900, take_profit=2_200)
    view = store.view({"BTC/USDT": 2_100})
    assert view.cash == pytest.approx(600)  # 2,000 notional / 5x = 400 margin
    assert view.equity == pytest.approx(1_100)  # 600 + 400 margin + 100 unrealized
    trade = store.close_position("BTC/USDT", 2_100, CloseReason.SIGNAL)
    assert trade.pnl == pytest.approx(100)
    assert store.view().cash == pytest.approx(1_100)


def test_decisions_and_equity_are_recorded(tmp_path) -> None:
    store = PortfolioStore(tmp_path / "d.sqlite", 10_000, clock=lambda: NOW)
    signal = technical()
    decision = _risk().decide(signal, sentiment(), portfolio(), now=NOW)
    store.record_decision(signal, decision, 0.1)
    store.record_equity()
    assert store.recent_decisions()[0]["action"] == "EXECUTE_BUY"
    assert store.equity_history()[0]["equity"] == pytest.approx(10_000)


# --------------------------------------------------------------- redaction


def test_redact_masks_tokens_in_urls_and_registered_secrets() -> None:
    register_secret("sup3r-secret-key")
    text = (
        "401, url='https://cryptopanic.com/api/v1/posts/?auth_token=abc123def&public=true' "
        "https://api.telegram.org/bot123456:ABCdef_ghi/sendMessage key=sup3r-secret-key"
    )
    out = redact(text)
    assert "abc123def" not in out and "ABCdef_ghi" not in out and "sup3r-secret-key" not in out
    assert "auth_token=***" in out and "&public=true" in out


def test_logging_filter_redacts_tracebacks() -> None:
    register_secret("trace-secret-999")
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed %s", ("trace-secret-999",), None)
    try:
        raise RuntimeError("boom trace-secret-999")
    except RuntimeError:
        import sys

        record.exc_info = sys.exc_info()
    RedactingFilter().filter(record)
    assert "trace-secret-999" not in record.getMessage()
    assert "trace-secret-999" not in (record.exc_text or "")


# --------------------------------------------------------------- sentiment


def test_negated_good_news_is_bearish_and_duplicates_count_once() -> None:
    score, bulls, bears = score_headlines(["SEC delays spot ETF approval"])
    assert bulls == 0 and bears >= 1 and score < 0
    once, _, _ = score_headlines(["Bitcoin rally continues"])
    twice, _, _ = score_headlines(["Bitcoin rally continues", "bitcoin  rally continues"])
    assert once == twice
