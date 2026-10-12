"""Strategy tournament: contenders, elimination, champion selection."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from agents.strategies import STRATEGY_NAMES, build_strategy
from core.arena import (
    ArenaStore,
    CachedCandleSource,
    _review_champion,
    arena_dir,
    current_champion,
    load_json,
    run_arena,
    state_path,
)
from core.models import neutral_sentiment
from tests.conftest import make_settings

STEP = 4 * 3600 * 1000


def _rows(count: int = 320, start: float = 100.0, drift: float = 0.002, last: float | None = None):
    now_ms = int(time.time() * 1000)
    first = (now_ms // STEP - count + 1) * STEP
    rows, price = [], start
    for i in range(count):
        price *= 1 + drift
        close = last if (last is not None and i >= count - 2) else price
        rows.append([first + i * STEP, close * 0.999, close * 1.004, close * 0.996, close, 10.0])
    return rows


class FakeSource:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls += 1
        return self.rows[-limit:]

    async def close(self):
        return None


def _settings(tmp_path, **extra):
    base = dict(
        paper_trading=True,
        data_dir=str(tmp_path),
        symbols="BTC/USDT",
        timeframe="4h",
        ohlcv_limit=300,
        starting_balance=1000,
        arena_enabled=True,
        arena_auto_select=True,
        arena_strategies="trend,rsi_dip",
        fear_greed_url="",
        rss_feeds="",
    )
    base.update(extra)
    return make_settings(**base)


def test_every_strategy_scores_a_frame() -> None:
    import pandas as pd

    rows = _rows(400)
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for name in STRATEGY_NAMES:
        result = build_strategy(name).evaluate(frame)
        assert result is not None, name
        assert 0.0 <= result.confidence <= 1.0


def test_arena_runs_each_contender_once_per_symbol_download(tmp_path) -> None:
    settings = _settings(tmp_path)
    source = FakeSource(_rows())
    cache = CachedCandleSource(source)
    state = asyncio.run(run_arena(settings, cache, neutral_sentiment("test")))
    assert source.calls == 1  # both contenders share one download
    assert {row["strategy"] for row in state["standings"]} == {"trend", "rsi_dip"}
    assert state["champion"] == "trend"  # no research yet: STRATEGY
    assert all(row["deposits"] == 1000 for row in state["standings"])
    assert load_json(state_path(settings))["champion"] == "trend"
    assert current_champion(settings) == "trend"


def test_research_picks_the_first_champion(tmp_path) -> None:
    settings = _settings(tmp_path)
    arena_dir(settings).mkdir(parents=True)
    (arena_dir(settings) / "research.json").write_text(json.dumps({"ranking": [{"strategy": "rsi_dip"}, {"strategy": "trend"}]}))
    assert current_champion(settings) == "rsi_dip"
    state = asyncio.run(run_arena(settings, CachedCandleSource(FakeSource(_rows())), neutral_sentiment("test")))
    assert state["champion"] == "rsi_dip"
    assert "backtest" in state["history"][0]["reason"]


def test_auto_select_off_keeps_strategy(tmp_path) -> None:
    settings = _settings(tmp_path, arena_auto_select=False)
    assert current_champion(settings) is None


def test_a_strategy_that_loses_its_limit_is_eliminated(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = ArenaStore(arena_dir(settings) / "trend.sqlite", 1000, fee_rate=0.0)
    store.open_position(symbol="BTC/USDT", side="long", size=9, price=100, stop_loss=10, take_profit=500)
    store.close()
    crash = _rows(start=100, drift=0.0, last=80.0)  # position worth 720 + 100 cash = 820: -18%
    state = asyncio.run(run_arena(settings, CachedCandleSource(FakeSource(crash)), neutral_sentiment("test")))
    trend = next(row for row in state["standings"] if row["strategy"] == "trend")
    assert trend["status"] == "eliminated"
    assert trend["open_positions"] == []
    assert state["champion"] == "rsi_dip"  # the eliminated champion is replaced at once
    # Eliminated strategies do not trade again.
    again = asyncio.run(run_arena(settings, CachedCandleSource(FakeSource(_rows())), neutral_sentiment("test")))
    assert next(r for r in again["standings"] if r["strategy"] == "trend")["status"] == "eliminated"


def _row(name, pct, trades, status="active"):
    return {"strategy": name, "true_profit_pct": pct, "trades": trades, "status": status}


def test_switch_needs_a_clear_lead_and_enough_trades(tmp_path) -> None:
    settings = _settings(tmp_path, arena_strategies="trend,rsi_dip,ema_cross")
    now = datetime(2026, 10, 20, 1, tzinfo=timezone.utc)
    state = {"champion": "trend"}
    # Ahead by 3 points but only 2 trades: no switch.
    assert _review_champion(settings, state, [_row("trend", 0.01, 4), _row("rsi_dip", 0.04, 2)], now, None) == "trend"
    # Next day, 3 trades and ahead by 3 points: switch.
    state.pop("last_review")
    assert _review_champion(settings, state, [_row("trend", 0.01, 4), _row("rsi_dip", 0.04, 3)], now + timedelta(days=1), None) == "rsi_dip"
    assert state["history"][-1]["from"] == "trend"
    # Ahead by only 1 point: stays.
    state.pop("last_review")
    assert _review_champion(settings, state, [_row("rsi_dip", 0.04, 3), _row("ema_cross", 0.05, 9)], now + timedelta(days=2), None) == "rsi_dip"


def test_review_happens_once_a_day(tmp_path) -> None:
    settings = _settings(tmp_path)
    now = datetime(2026, 10, 20, 1, tzinfo=timezone.utc)
    state = {"champion": "trend"}
    _review_champion(settings, state, [_row("trend", 0.0, 0), _row("rsi_dip", 0.0, 0)], now, None)
    later = now + timedelta(hours=5)
    # Same day: even a big lead does not switch until tomorrow's review.
    assert _review_champion(settings, state, [_row("trend", 0.0, 5), _row("rsi_dip", 0.5, 5)], later, None) == "trend"


def test_dashboard_shows_the_tournament(tmp_path) -> None:
    from core.control import Controller
    from core.state import PortfolioStore
    from dashboard.server import DashboardContext, build_state

    settings = _settings(tmp_path)
    asyncio.run(run_arena(settings, CachedCandleSource(FakeSource(_rows())), neutral_sentiment("test")))
    store = PortfolioStore(tmp_path / "portfolio.sqlite", 1000, fee_rate=0.0)
    state = build_state(DashboardContext(settings, store, None, Controller("paper")))
    assert state["arena"]["champion"] == "trend"
    assert len(state["arena"]["standings"]) == 2
    assert state["config"]["strategy"] == "trend"
    store.close()


def test_arena_is_refused_in_live_mode(tmp_path) -> None:
    with pytest.raises(Exception):
        make_settings(paper_trading=False, live_trading_confirm=True, exchange_api_key="k", exchange_api_secret="s",
                      arena_enabled=True, data_dir=str(tmp_path))


def test_once_run_runs_the_tournament_and_exports_it(tmp_path, monkeypatch) -> None:
    import main as entry
    from agents.execution import ExecutionAgent
    from agents.market_data import MarketDataAgent
    from agents.risk import RiskManager
    from core.state import PortfolioStore
    from tests.test_orchestrator import RecordingNotifier, ScriptedSentiment
    from tests.conftest import sentiment

    for key, value in {
        "PAPER_TRADING": "true", "DATA_DIR": str(tmp_path / "data"), "LOG_DIR": str(tmp_path / "logs"),
        "STARTING_BALANCE": "1000", "SYMBOLS": "BTC/USDT", "DASHBOARD_ENABLED": "false", "TIMEFRAME": "4h",
        "ARENA_ENABLED": "true", "ARENA_AUTO_SELECT": "true", "ARENA_STRATEGIES": "trend,ema_cross",
    }.items():
        monkeypatch.setenv(key, value)

    def fake_runtime(settings):
        store = PortfolioStore(f"{settings.data_dir}/portfolio.sqlite", settings.starting_balance, fee_rate=0.0)
        market = MarketDataAgent(CachedCandleSource(FakeSource(_rows())), "4h", 300, strategy=build_strategy(entry.active_strategy(settings)))
        return entry.Runtime(settings, store, market, ScriptedSentiment(sentiment()), RiskManager(settings),
                             ExecutionAgent(settings, store), RecordingNotifier())

    monkeypatch.setattr(entry, "build_runtime", fake_runtime)
    out = tmp_path / "site" / "state.json"
    assert asyncio.run(entry.amain(["--once", "--export-state", str(out)])) == 0
    state = json.loads(out.read_text(encoding="utf-8"))
    assert state["arena"]["champion"] == "trend"
    assert {r["strategy"] for r in state["arena"]["standings"]} == {"trend", "ema_cross"}
