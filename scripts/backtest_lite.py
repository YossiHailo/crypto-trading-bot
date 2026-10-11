"""Walk a synthetic price path through the same agents used in production.

This is a dry pipeline check (no network), not evidence of an edge. For a
real historical test with benchmarks use ``python main.py backtest``. Sentiment is held at
NEUTRAL so only the technical rules and the paper account are exercised.
No network calls are made.

Usage:

    python scripts/backtest_lite.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.execution import ExecutionAgent  # noqa: E402
from agents.market_data import MarketDataAgent  # noqa: E402
from agents.risk import RiskManager  # noqa: E402
from config.settings import Settings  # noqa: E402
from core.models import neutral_sentiment  # noqa: E402
from core.orchestrator import run_cycle  # noqa: E402
from core.state import PortfolioStore  # noqa: E402
from notifications.telegram import TelegramNotifier  # noqa: E402


def synthetic_ohlcv(n: int, seed: int = 7, start: float = 100.0) -> list[list[float]]:
    """Sine wave plus a little noise so indicators actually change bar to bar."""
    rng = np.random.default_rng(seed)
    step = np.arange(n, dtype="float64")
    close = start + 6.0 * np.sin(step / 10.0) + np.cumsum(rng.normal(0.0, 0.25, size=n))
    close = np.maximum(close, 1.0)
    open_ = np.empty(n, dtype="float64")
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = rng.uniform(0.001, 0.01, size=n)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.uniform(10.0, 100.0, size=n)
    rows: list[list[float]] = []
    for index in range(n):
        rows.append(
            [
                1_700_000_000_000 + index * 3_600_000,
                float(open_[index]),
                float(high[index]),
                float(low[index]),
                float(close[index]),
                float(volume[index]),
            ]
        )
    return rows


class ReplayCandleSource:
    """Serves a growing prefix of one precomputed series."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.end = len(rows)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        del symbol, timeframe
        return self.rows[: self.end][-limit:]

    async def close(self) -> None:
        return None


class FlatSentiment:
    async def evaluate(self):
        return neutral_sentiment("Backtest holds sentiment at NEUTRAL.")

    async def close(self) -> None:
        return None


async def run_backtest(
    *,
    bars: int = 180,
    seed: int = 7,
    min_history: int = 80,
    data_dir: str | Path,
    settings: Settings | None = None,
) -> dict[str, float | int]:
    """Replay ``bars`` candles and return ending equity plus trade counts."""
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    settings = settings or Settings(
        _env_file=None,
        symbols="SYN/USDT",
        paper_trading=True,
        data_dir=str(directory),
        log_dir=str(directory / "logs"),
        min_confidence=0.45,
        min_notional=1,
        strategy="legacy",
        ohlcv_limit=70,
        dashboard_enabled=False,
    )
    store = PortfolioStore(
        directory / "backtest.sqlite",
        settings.starting_balance,
        quote_currency=settings.quote_currency,
        fee_rate=settings.fee_rate,
    )
    source = ReplayCandleSource(synthetic_ohlcv(bars, seed=seed))
    from agents.strategies import build_strategy

    market = MarketDataAgent(source, settings.timeframe, settings.ohlcv_limit, strategy=build_strategy(settings.strategy))
    execution = ExecutionAgent(settings, store)
    notifier = TelegramNotifier("", "")
    risk = RiskManager(settings)
    sentiment = FlatSentiment()
    cycles = 0
    try:
        for end in range(min_history, bars + 1):
            source.end = end
            await run_cycle(
                symbols=settings.symbol_list(),
                market=market,
                sentiment_agent=sentiment,
                risk=risk,
                execution=execution,
                notifier=notifier,
                settings=settings,
            )
            cycles += 1
        view = store.view()
        closed = store._conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()["n"]
        return {
            "cycles": cycles,
            "equity": view.equity,
            "cash": view.cash,
            "open_positions": view.open_position_count,
            "closed_trades": int(closed),
        }
    finally:
        await execution.close()
        await notifier.close()


def main() -> None:
    result = asyncio.run(run_backtest(data_dir=Path("data") / "backtest"))
    print("Backtest-lite finished (synthetic data, not a performance claim).")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
