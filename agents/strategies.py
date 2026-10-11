"""Signal strategies. Each one scores a frame of *closed* candles.

The same strategy object is used by the live loop, the paper loop and the
historical backtester, so a backtest measures the code that will trade.

``trend``  (default) Donchian breakout in the direction of a 50/200 EMA trend,
           exit on a 10-bar opposite channel break. One philosophy: follow
           trends, cut losers with the ATR stop, let the channel trail winners.
``legacy`` The original weighted mix of RSI, MACD, SMA, Bollinger and EMA.
           Kept so the backtester can compare it. Note its known weaknesses:
           mean-reversion (RSI, Bollinger) and trend (MACD, SMA, EMA) parts
           pull against each other, and the EMA12/EMA26 component is the sign
           of the MACD line, so that information is counted twice.

Neither strategy has a demonstrated edge. Run ``python main.py backtest``
on real history before trusting either one with money.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from core.indicators import atr as atr_series
from core.indicators import donchian, ema
from core.models import SignalAction


@dataclass(frozen=True)
class StrategyResult:
    action: SignalAction
    confidence: float
    reasoning: str
    atr: float
    indicators: dict[str, float] = field(default_factory=dict)
    exit_long: bool = False
    exit_short: bool = False


class Strategy(Protocol):
    name: str
    min_bars: int

    def evaluate(self, frame: pd.DataFrame) -> StrategyResult | None:
        """Score the last row of ``frame`` (closed bars only). ``None`` means not enough data."""


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class TrendBreakoutStrategy:
    """Donchian breakout filtered by the 50/200 EMA trend."""

    name = "trend"

    def __init__(
        self,
        *,
        fast: int = 50,
        slow: int = 200,
        entry_channel: int = 20,
        exit_channel: int = 10,
        atr_period: int = 14,
    ) -> None:
        if not fast < slow:
            raise ValueError("fast EMA must be shorter than slow EMA")
        self.fast = fast
        self.slow = slow
        self.entry_channel = entry_channel
        self.exit_channel = exit_channel
        self.atr_period = atr_period
        self.min_bars = slow + 10

    def evaluate(self, frame: pd.DataFrame) -> StrategyResult | None:
        if len(frame) < self.min_bars:
            return None
        close = frame["close"].astype("float64")
        ema_fast = ema(close, self.fast)
        ema_slow = ema(close, self.slow)
        upper, lower = donchian(frame, self.entry_channel)
        exit_upper, exit_lower = donchian(frame, self.exit_channel)
        atr_values = atr_series(frame, self.atr_period)

        values = {
            "close": close.iloc[-1],
            "ema_fast": ema_fast.iloc[-1],
            "ema_slow": ema_slow.iloc[-1],
            "upper": upper.iloc[-1],
            "lower": lower.iloc[-1],
            "exit_upper": exit_upper.iloc[-1],
            "exit_lower": exit_lower.iloc[-1],
            "atr": atr_values.iloc[-1],
        }
        if not all(_finite(value) for value in values.values()):
            return None
        price = float(values["close"])
        fast_v = float(values["ema_fast"])
        slow_v = float(values["ema_slow"])
        up = float(values["upper"])
        low = float(values["lower"])
        atr_v = max(0.0, float(values["atr"]))
        indicators = {key: float(value) for key, value in values.items()}

        uptrend = fast_v > slow_v and price > slow_v
        downtrend = fast_v < slow_v and price < slow_v
        exit_long = price < float(values["exit_lower"])
        exit_short = price > float(values["exit_upper"])
        trend_text = "uptrend" if uptrend else "downtrend" if downtrend else "no clear trend"
        base = (
            f"{trend_text} (EMA{self.fast} {fast_v:.8g} vs EMA{self.slow} {slow_v:.8g}); "
            f"close {price:.8g}; {self.entry_channel}-bar channel {low:.8g}..{up:.8g}"
        )

        def strength(distance: float, spread: float) -> float:
            if atr_v <= 0:
                return 0.6
            return min(1.0, 0.6 + 0.25 * _clamp01(distance / atr_v) + 0.15 * _clamp01(spread / (2 * atr_v)))

        if uptrend and price > up:
            confidence = strength(price - up, fast_v - slow_v)
            return StrategyResult(
                SignalAction.BUY,
                confidence,
                f"BUY breakout above the channel in an {base}",
                atr_v,
                indicators,
                exit_short=True,
            )
        if downtrend and price < low:
            confidence = strength(low - price, slow_v - fast_v)
            return StrategyResult(
                SignalAction.SELL,
                confidence,
                f"SELL breakdown below the channel in a {base}",
                atr_v,
                indicators,
                exit_long=True,
            )
        notes = []
        if exit_long:
            notes.append(f"close below the {self.exit_channel}-bar low: exit longs")
        if exit_short:
            notes.append(f"close above the {self.exit_channel}-bar high: exit shorts")
        reason = "HOLD. " + base + (". " + "; ".join(notes) if notes else "")
        return StrategyResult(
            SignalAction.HOLD,
            0.0,
            reason,
            atr_v,
            indicators,
            exit_long=exit_long,
            exit_short=exit_short,
        )


class LegacyScoreStrategy:
    """The original weighted indicator score (see the module docstring)."""

    name = "legacy"

    def __init__(self) -> None:
        from core.indicators import MIN_BARS

        self.min_bars = MIN_BARS

    def evaluate(self, frame: pd.DataFrame) -> StrategyResult | None:
        from agents.market_data import interpret, snapshot_from_frame

        snapshot = snapshot_from_frame(frame)
        if snapshot is None:
            return None
        action, confidence, reasoning, raw = interpret(snapshot)
        indicators = {
            "rsi": snapshot.rsi,
            "macd": snapshot.macd,
            "macd_signal": snapshot.macd_signal,
            "macd_hist": snapshot.macd_hist,
            "sma_fast": snapshot.sma_fast,
            "sma_slow": snapshot.sma_slow,
            "ema_fast": snapshot.ema_fast,
            "ema_slow": snapshot.ema_slow,
            "bb_upper": snapshot.bb_upper,
            "bb_mid": snapshot.bb_mid,
            "bb_lower": snapshot.bb_lower,
            "atr": snapshot.atr,
            "score": raw,
        }
        return StrategyResult(
            action,
            confidence,
            reasoning,
            snapshot.atr,
            indicators,
            exit_long=action == SignalAction.SELL,
            exit_short=action == SignalAction.BUY,
        )


STRATEGY_NAMES = ("trend", "legacy")


def build_strategy(name: str) -> Strategy:
    if name == "trend":
        return TrendBreakoutStrategy()
    if name == "legacy":
        return LegacyScoreStrategy()
    raise ValueError(f"Unknown strategy {name!r}")
