"""Preview the dashboard with made-up data. No exchange, no network, no orders.

    python scripts/demo_dashboard.py            # then open http://127.0.0.1:8765

The demo writes to data/demo/ and never touches your real portfolio file.
Buttons work against the demo account (close-all closes the demo position).
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.execution import ExecutionAgent  # noqa: E402
from config.settings import Settings  # noqa: E402
from core.control import Controller  # noqa: E402
from core.models import (  # noqa: E402
    CloseReason,
    CycleReport,
    RiskAction,
    RiskDecision,
    SentimentBias,
    SentimentSignal,
    SignalAction,
    TechnicalSignal,
)
from core.state import PortfolioStore  # noqa: E402
from dashboard.server import DashboardContext, DashboardServer  # noqa: E402


def build_demo(data_dir: Path, port: int) -> tuple[Settings, PortfolioStore, ExecutionAgent, Controller]:
    shutil.rmtree(data_dir, ignore_errors=True)
    clock = {"now": datetime.now(timezone.utc) - timedelta(days=12)}
    settings = Settings(
        _env_file=None,
        paper_trading=True,
        data_dir=str(data_dir),
        dashboard_port=port,
        symbols="BTC/USDT,ETH/USDT,SOL/USDT",
    )
    store = PortfolioStore(
        data_dir / "portfolio.sqlite",
        settings.starting_balance,
        fee_rate=settings.fee_rate,
        clock=lambda: clock["now"],
        max_drawdown_pct=settings.max_drawdown_pct,
    )
    rng = random.Random(4)
    btc, eth = 62_000.0, 2_450.0
    trades = [
        ("ETH/USDT", "long", 2.0, 2_380.0, 2_452.0, CloseReason.TAKE_PROFIT),
        ("BTC/USDT", "long", 0.05, 61_200.0, 60_350.0, CloseReason.STOP_LOSS),
        ("SOL/USDT", "long", 18.0, 141.0, 147.5, CloseReason.SIGNAL),
    ]
    for step in range(12 * 24):
        clock["now"] += timedelta(hours=1)
        btc *= math.exp(rng.gauss(0.0004, 0.006))
        eth *= math.exp(rng.gauss(0.0003, 0.008))
        if step in (40, 110, 200) and trades:
            symbol, side, size, entry, exit_price, reason = trades.pop(0)
            store.open_position(symbol=symbol, side=side, size=size, price=entry, stop_loss=entry * 0.97, take_profit=entry * 1.03)
            store.close_position(symbol, exit_price, reason)
        if step == 250:
            store.open_position(
                symbol="BTC/USDT", side="long", size=0.06, price=btc, stop_loss=btc * 0.965, take_profit=btc * 1.07,
                exchange_protected=True,
            )
        store.record_equity({"BTC/USDT": btc, "ETH/USDT": eth})
    store.add_event("info", "Bot started in paper mode.")
    store.add_event("warning", "SOL/USDT: market data failed: binance GET /api/v3/klines 451 (demo)")
    store.add_event("info", f"Opened EXECUTE_BUY BTC/USDT @ {btc:.2f}")

    prices = {"BTC/USDT": btc, "ETH/USDT": eth, "SOL/USDT": 151.2}
    def trend_ind(close: float, fast: float, slow: float, upper: float, lower: float) -> dict[str, float]:
        return {"close": close, "ema_fast": fast, "ema_slow": slow, "upper": upper, "lower": lower,
                "exit_upper": upper * 0.99, "exit_lower": lower * 1.01, "atr": close * 0.012}

    signals = [
        TechnicalSignal("BTC/USDT", SignalAction.HOLD, 0.0, "HOLD. uptrend (EMA50 above EMA200); close inside the 20-bar channel", btc * 0.012, btc, clock["now"],
                        indicators=trend_ind(btc * 0.998, btc * 0.97, btc * 0.9, btc * 1.019, btc * 0.95), strategy="trend"),
        TechnicalSignal("ETH/USDT", SignalAction.BUY, 0.71, "BUY breakout above the channel in an uptrend (EMA50 above EMA200)", eth * 0.015, eth, clock["now"],
                        indicators=trend_ind(eth, eth * 0.96, eth * 0.9, eth * 0.99, eth * 0.9), strategy="trend"),
        TechnicalSignal("SOL/USDT", SignalAction.HOLD, 0.0, "HOLD. no clear trend; close below the 10-bar low: exit longs", 3.1, 151.2, clock["now"],
                        indicators=trend_ind(151.2, 150.0, 152.0, 160.0, 149.0), exit_long=True, strategy="trend"),
    ]
    sentiment = SentimentSignal(SentimentBias.NEUTRAL, 0.08, "NEUTRAL score +0.08. Fear & Greed 58 (Greed) -> +0.16. Headlines 30 items, bullish hits 3, bearish hits 3 -> +0.00", fear_greed=58, sources_used=("fear_greed", "headlines"))
    for signal in signals:
        action = RiskAction.NO_ACTION
        reason = "Technical signal is HOLD."
        if signal.symbol == "ETH/USDT":
            reason = "Combined confidence 0.52 is below the minimum 0.55. Sentiment NEUTRAL (+0.08) alignment +0.08."
        decision = RiskDecision(action, signal.symbol, 0.0, signal.price, 0.0, 0.0, signal.confidence, reason, timestamp=clock["now"])
        store.record_decision(signal, decision, sentiment.score)
    settings_live = settings
    execution = ExecutionAgent(settings_live, store)
    controller = Controller("paper")
    report = CycleReport(started_at=clock["now"] - timedelta(seconds=4), finished_at=clock["now"], signals=signals, sentiment=sentiment)
    controller.record_report(report, prices)
    controller.status.next_cycle_at = datetime.now(timezone.utc) + timedelta(minutes=9)
    clock["now"] = datetime.now(timezone.utc)
    controller.status.last_cycle_finished = clock["now"] - timedelta(minutes=6)
    return settings, store, execution, controller


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    settings, store, execution, controller = build_demo(ROOT / "data" / "demo", args.port)
    server = DashboardServer(DashboardContext(settings, store, execution, controller))
    await server.start()
    print(f"Demo dashboard: http://127.0.0.1:{server.port}  (Ctrl+C to stop)")
    try:
        await controller.stop_event.wait()
    finally:
        await server.close()
        await execution.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
