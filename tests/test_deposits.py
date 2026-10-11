"""Simulated monthly deposits (paper mode) and the remote-control helpers for --once."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from agents.market_data import FallbackCandleSource
from core.control import Controller
from core.state import PortfolioStore, next_deposit_at
from dashboard.server import DashboardContext, build_state
from tests.conftest import make_settings


def _store(tmp_path, clock, balance=1_000.0):
    return PortfolioStore(tmp_path / "d.sqlite", balance, fee_rate=0.0, clock=lambda: clock["now"], max_drawdown_pct=0.2)


def _at(y, m, d, h=0, minute=0):
    return datetime(y, m, d, h, minute, tzinfo=timezone.utc)


def test_initial_balance_is_the_first_deposit(tmp_path) -> None:
    clock = {"now": _at(2026, 10, 8, 12)}
    store = _store(tmp_path, clock)
    assert store.total_deposits() == 1_000
    assert [d["note"] for d in store.deposits()] == ["initial"]
    # Same day, nothing due yet.
    assert store.apply_monthly_deposits(100, 1) == []


def test_monthly_deposit_on_the_first_adds_cash_and_moves_baselines(tmp_path) -> None:
    clock = {"now": _at(2026, 10, 8, 12)}
    store = _store(tmp_path, clock)
    clock["now"] = _at(2026, 11, 1, 0, 7)
    applied = store.apply_monthly_deposits(100, 1)
    assert [when for when, _ in applied] == [_at(2026, 11, 1)]
    view = store.view()
    assert view.cash == pytest.approx(1_100)
    assert view.day_start_equity == pytest.approx(1_100)  # a deposit is not a daily gain
    assert view.peak_equity == pytest.approx(1_100)  # nor a new high for the kill switch
    assert store.total_deposits() == pytest.approx(1_100)
    # Running again in the same month does not deposit twice.
    clock["now"] = _at(2026, 11, 20)
    assert store.apply_monthly_deposits(100, 1) == []
    assert store.view().cash == pytest.approx(1_100)


def test_missed_deposits_are_caught_up_each_with_its_due_date(tmp_path) -> None:
    clock = {"now": _at(2026, 10, 8)}
    store = _store(tmp_path, clock)
    clock["now"] = _at(2027, 1, 15)  # bot did not run for months
    applied = store.apply_monthly_deposits(100, 1)
    assert [when.date().isoformat() for when, _ in applied] == ["2026-11-01", "2026-12-01", "2027-01-01"]
    assert store.total_deposits() == pytest.approx(1_300)


def test_day_31_is_clamped_to_the_last_day_of_short_months(tmp_path) -> None:
    clock = {"now": _at(2027, 1, 31, 12)}
    store = _store(tmp_path, clock)
    clock["now"] = _at(2027, 3, 1)
    applied = store.apply_monthly_deposits(50, 31)
    assert [when.date().isoformat() for when, _ in applied] == ["2027-02-28"]
    assert next_deposit_at(_at(2027, 3, 1), 31) == _at(2027, 3, 31)
    assert next_deposit_at(_at(2026, 12, 2), 1) == _at(2027, 1, 1)


def test_deposit_does_not_trip_the_drawdown_kill_switch(tmp_path) -> None:
    clock = {"now": _at(2026, 10, 8)}
    store = _store(tmp_path, clock)
    store.open_position(symbol="BTC/USDT", side="long", size=10, price=90, stop_loss=50, take_profit=200)
    clock["now"] = _at(2026, 11, 1, 1)
    store.apply_monthly_deposits(100, 1)
    # Equity 1,100 at entry price; a 15% drop in BTC is a ~12% account loss: below the 20% limit.
    assert store.update_peak_and_check_drawdown({"BTC/USDT": 76.5}) is None


def test_dashboard_reports_true_profit_against_money_put_in(tmp_path) -> None:
    clock = {"now": _at(2026, 10, 8)}
    store = _store(tmp_path, clock)
    clock["now"] = _at(2026, 11, 2)
    store.apply_monthly_deposits(100, 1)
    settings = make_settings(monthly_deposit=100, deposit_day=1, starting_balance=1_000, data_dir=str(tmp_path))
    state = build_state(DashboardContext(settings, store, None, Controller("paper")))
    account = state["account"]
    assert account["total_deposits"] == pytest.approx(1_100)
    assert account["total_change"] == pytest.approx(0.0)  # 1,100 in, 1,100 worth: no profit yet
    assert account["next_deposit_at"] == _at(2026, 12, 1)
    assert [d["note"] for d in state["deposits"]] == ["initial", "monthly"]
    json.dumps(state, default=str)


def test_deposits_are_refused_in_live_mode(tmp_path) -> None:
    with pytest.raises(Exception):
        make_settings(monthly_deposit=100, paper_trading=False, live_trading_confirm=True,
                      exchange_api_key="k", exchange_api_secret="s", data_dir=str(tmp_path))


class _Source:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.closed = rows, error, False

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        if self.error:
            raise RuntimeError(self.error)
        return self.rows

    async def close(self):
        self.closed = True


def test_fallback_candle_source_uses_the_next_exchange() -> None:
    blocked = _Source(error="451 restricted location")
    backup = _Source(rows=[[1, 2, 3, 1, 2, 5]])
    source = FallbackCandleSource([blocked, backup])
    assert asyncio.run(source.fetch_ohlcv("BTC/USDT", "1h", 10)) == [[1, 2, 3, 1, 2, 5]]
    asyncio.run(source.close())
    assert blocked.closed and backup.closed
    failing = FallbackCandleSource([_Source(error="a"), _Source(rows=[])])
    with pytest.raises(RuntimeError, match="all candle sources failed"):
        asyncio.run(failing.fetch_ohlcv("BTC/USDT", "1h", 10))


def test_once_run_applies_remote_action_and_writes_a_static_snapshot(tmp_path, monkeypatch) -> None:
    import main as entry
    from agents.execution import ExecutionAgent
    from agents.risk import RiskManager
    from core.models import SignalAction
    from tests.conftest import sentiment, technical
    from tests.test_orchestrator import RecordingNotifier, ScriptedMarket, ScriptedSentiment

    for key, value in {
        "PAPER_TRADING": "true", "DATA_DIR": str(tmp_path / "data"), "LOG_DIR": str(tmp_path / "logs"),
        "STARTING_BALANCE": "1000", "MONTHLY_DEPOSIT": "100", "SYMBOLS": "BTC/USDT", "DASHBOARD_ENABLED": "false",
    }.items():
        monkeypatch.setenv(key, value)

    def fake_runtime(settings):
        store = PortfolioStore(f"{settings.data_dir}/portfolio.sqlite", settings.starting_balance, fee_rate=0.0)
        market = ScriptedMarket({"BTC/USDT": [technical(action=SignalAction.HOLD, confidence=0.0, price=100.0)]})
        return entry.Runtime(settings, store, market, ScriptedSentiment(sentiment()), RiskManager(settings),
                             ExecutionAgent(settings, store), RecordingNotifier())

    monkeypatch.setattr(entry, "build_runtime", fake_runtime)
    out = tmp_path / "site" / "state.json"
    assert asyncio.run(entry.amain(["--once", "--action", "pause", "--export-state", str(out)])) == 0
    state = json.loads(out.read_text(encoding="utf-8"))
    assert state["static"] is True and state["status"]["paused"] is True
    assert state["account"]["total_deposits"] == pytest.approx(1_000)
    assert state["account"]["monthly_deposit"] == 100 and state["account"]["next_deposit_at"]
    assert state["signals"][0]["symbol"] == "BTC/USDT"
    with pytest.raises(SystemExit):
        entry.parse_args(["--action", "pause"])  # needs --once
