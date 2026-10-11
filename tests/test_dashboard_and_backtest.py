"""Dashboard HTTP API (real sockets) and the backtest engine's timing rules."""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from agents.execution import ExecutionAgent
from agents.strategies import StrategyResult, TrendBreakoutStrategy
from backtest.engine import buy_and_hold, compute_metrics, period_breakdown, random_entries, run_strategy
from core.control import Controller
from core.models import SignalAction
from core.state import PortfolioStore
from dashboard.server import DashboardContext, DashboardServer
from tests.conftest import make_settings

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _http(port: int, method: str, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    lines = [f"{method} {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Content-Length: 0"]
    lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
    await writer.drain()
    raw = await reader.read()
    writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), body


async def _server(tmp_path, **overrides):
    settings = make_settings(paper_trading=True, data_dir=str(tmp_path), dashboard_port=_free_port(), **overrides)
    store = PortfolioStore(tmp_path / "dash.sqlite", 10_000, clock=lambda: NOW)
    execution = ExecutionAgent(settings, store)
    controller = Controller("paper")
    server = DashboardServer(DashboardContext(settings, store, execution, controller))
    await server.start()
    return server, store, execution, controller


@pytest.mark.asyncio
async def test_dashboard_serves_state_and_page(tmp_path) -> None:
    server, store, execution, _ = await _server(tmp_path)
    try:
        status, body = await _http(server.port, "GET", "/")
        assert status == 200 and b"<html" in body
        status, body = await _http(server.port, "GET", "/api/state")
        state = json.loads(body)
        assert status == 200
        assert state["account"]["equity"] == pytest.approx(10_000)
        assert state["config"]["mode"] == "paper"
        assert "exchange_api_secret" not in json.dumps(state)
    finally:
        await server.close()
        await execution.close()


@pytest.mark.asyncio
async def test_dashboard_controls_need_the_csrf_header(tmp_path) -> None:
    server, store, execution, controller = await _server(tmp_path)
    try:
        status, _ = await _http(server.port, "POST", "/api/control/pause")
        assert status == 403
        assert store.view().paused is False
        status, body = await _http(server.port, "POST", "/api/control/pause", {"X-Requested-With": "ctb-dashboard"})
        assert status == 200 and json.loads(body)["ok"]
        assert store.view().paused is True
        status, _ = await _http(server.port, "POST", "/api/control/stop", {"X-Requested-With": "ctb-dashboard"})
        assert status == 200 and controller.stop_event.is_set()
        status, _ = await _http(server.port, "GET", "/api/control/pause")
        assert status == 405
    finally:
        await server.close()
        await execution.close()


@pytest.mark.asyncio
async def test_dashboard_token_and_host_checks(tmp_path) -> None:
    server, store, execution, _ = await _server(tmp_path, dashboard_token="t0ken-abcdef")
    try:
        status, _ = await _http(server.port, "GET", "/api/state")
        assert status == 403
        status, _ = await _http(server.port, "GET", "/api/state", {"X-Dashboard-Token": "t0ken-abcdef"})
        assert status == 200
        # DNS rebinding: a foreign Host header is refused on a loopback bind.
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        writer.write(b"GET /api/state HTTP/1.1\r\nHost: evil.example\r\nX-Dashboard-Token: t0ken-abcdef\r\n\r\n")
        await writer.drain()
        raw = await reader.read()
        writer.close()
        assert b" 403 " in raw.split(b"\r\n")[0] + b" "
    finally:
        await server.close()
        await execution.close()


@pytest.mark.asyncio
async def test_dashboard_close_all(tmp_path) -> None:
    server, store, execution, controller = await _server(tmp_path)
    store.open_position(symbol="BTC/USDT", side="long", size=1, price=100, stop_loss=90, take_profit=120)
    controller.status.last_prices = {"BTC/USDT": 105.0}
    try:
        status, body = await _http(server.port, "POST", "/api/control/close-all", {"X-Requested-With": "ctb-dashboard"})
        assert status == 200
        assert "1 position(s) closed" in json.loads(body)["message"]
        assert store.get_position("BTC/USDT") is None
    finally:
        await server.close()
        await execution.close()


# ----------------------------------------------------------------- backtest


def _frame(closes, opens=None, highs=None, lows=None, step=4 * 3_600_000):
    n = len(closes)
    opens = opens if opens is not None else closes
    return pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype="int64") * step + 1_600_000_000_000,
            "open": opens,
            "high": highs if highs is not None else np.maximum(opens, closes) * 1.001,
            "low": lows if lows is not None else np.minimum(opens, closes) * 0.999,
            "close": closes,
            "volume": np.ones(n),
        }
    )


class _BuyOnce:
    """Says BUY exactly on the bar whose close equals ``trigger``."""

    name = "buy_once"
    min_bars = 5

    def __init__(self, trigger: float) -> None:
        self.trigger = trigger

    def evaluate(self, frame):
        last = float(frame["close"].iloc[-1])
        if abs(last - self.trigger) < 1e-9:
            return StrategyResult(SignalAction.BUY, 1.0, "test", atr=2.0)
        return StrategyResult(SignalAction.HOLD, 0.0, "test", atr=2.0)


def _bt_settings(**overrides):
    defaults = dict(
        ohlcv_limit=70, strategy="legacy", fee_rate=0.0, slippage_pct=0.001, risk_per_trade=0.01,
        atr_multiplier=1.5, reward_risk_ratio=2.0, min_reward_risk=1.0, min_notional=0.0, timeframe="4h",
    )
    defaults.update(overrides)
    return make_settings(**defaults)


def test_backtest_fills_at_the_next_open_not_the_signal_close() -> None:
    closes = np.linspace(100, 110, 120)
    opens = closes - 0.05
    trigger = float(closes[80])
    frame = _frame(closes, opens=opens)
    result = run_strategy(frame, _bt_settings(), _BuyOnce(trigger), start_index=59)
    assert result.trades, "expected the entry to be taken"
    trade = result.trades[0]
    assert trade.entry == pytest.approx(opens[81] * 1.001)  # next open plus slippage
    assert trade.entry_time == datetime.fromtimestamp(frame["timestamp"][81] / 1000, tz=timezone.utc)


def test_backtest_assumes_stop_before_target_inside_one_bar() -> None:
    closes = np.full(100, 100.0)
    highs = closes * 1.001
    lows = closes * 0.999
    highs[71] = 120.0  # bar after entry touches the target...
    lows[71] = 80.0  # ...and the stop
    frame = _frame(closes, highs=highs, lows=lows)
    result = run_strategy(frame, _bt_settings(slippage_pct=0.0), _BuyOnce(100.0), start_index=69)
    assert result.trades[0].reason == "stop_loss"


def test_metrics_on_a_known_curve() -> None:
    equity = np.array([100.0, 120.0, 90.0, 110.0])
    times = [datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 1, tzinfo=timezone.utc),
             datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2027, 1, 1, tzinfo=timezone.utc)]
    metrics = compute_metrics(equity, times, [], np.array([True, False, True, False]), "1d")
    assert metrics.total_return == pytest.approx(0.10)
    assert metrics.max_drawdown == pytest.approx(0.25)
    assert metrics.exposure == pytest.approx(0.5)


def test_full_pipeline_with_benchmarks_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 700)))
    frame = _frame(closes)
    settings = _bt_settings(ohlcv_limit=230, strategy="trend")
    strategy = TrendBreakoutStrategy()
    result = run_strategy(frame, settings, strategy, start_index=229)
    hold = buy_and_hold(frame, settings, 229)
    first = random_entries(frame, settings, 229, result, runs=20, seed=1)
    second = random_entries(frame, settings, 229, result, runs=20, seed=1)
    assert np.allclose(first.total_returns, second.total_returns)
    assert len(first.band_mid) == len(hold.equity) == len(result.equity)
    rows = period_breakdown(result, hold, 3)
    assert len(rows) == 3
