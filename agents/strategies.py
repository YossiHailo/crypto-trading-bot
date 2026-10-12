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
from core.indicators import bollinger_bands, donchian, ema, macd, rsi
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


# ------------------------------------------------------------------------
# Strategies for the strategy tournament (long only: they buy and later sell;
# they never short). Each one is a different market idea, so the tournament
# compares philosophies rather than tiny parameter tweaks.


def _last(series: pd.Series, offset: int = 1) -> float:
    return float(series.iloc[-offset])


class TrendFastStrategy(TrendBreakoutStrategy):
    """The trend strategy with shorter windows: reacts sooner, trades more, more false starts."""

    name = "trend_fast"

    def __init__(self) -> None:
        super().__init__(fast=20, slow=100, entry_channel=10, exit_channel=5)


class EmaCrossStrategy:
    """Buy when EMA20 crosses above EMA50 while the price is above EMA200; sell on the opposite cross."""

    name = "ema_cross"

    def __init__(self, fast: int = 20, slow: int = 50, filter_period: int = 200, atr_period: int = 14) -> None:
        self.fast, self.slow, self.filter_period, self.atr_period = fast, slow, filter_period, atr_period
        self.min_bars = filter_period + 10

    def evaluate(self, frame: pd.DataFrame) -> StrategyResult | None:
        if len(frame) < self.min_bars:
            return None
        close = frame["close"].astype("float64")
        fast, slow, flt = ema(close, self.fast), ema(close, self.slow), ema(close, self.filter_period)
        atr_v = _last(atr_series(frame, self.atr_period))
        values = [_last(fast), _last(fast, 2), _last(slow), _last(slow, 2), _last(flt), atr_v, _last(close)]
        if not all(_finite(v) for v in values):
            return None
        f_now, f_prev, s_now, s_prev, flt_v, atr_v, price = values
        indicators = {"close": price, "ema_fast": f_now, "ema_slow": s_now, "ema_filter": flt_v, "atr": atr_v}
        crossed_up = f_prev <= s_prev and f_now > s_now
        exit_long = f_now < s_now
        base = f"EMA{self.fast} {f_now:.8g} vs EMA{self.slow} {s_now:.8g}; EMA{self.filter_period} {flt_v:.8g}; close {price:.8g}"
        if crossed_up and price > flt_v:
            gap = (f_now - s_now) / atr_v if atr_v > 0 else 0.0
            return StrategyResult(SignalAction.BUY, min(1.0, 0.65 + 0.35 * _clamp01(gap)), f"BUY: EMA{self.fast} crossed above EMA{self.slow} above the long-term trend. {base}", atr_v, indicators)
        note = f"EMA{self.fast} below EMA{self.slow}: exit longs. " if exit_long else ""
        return StrategyResult(SignalAction.HOLD, 0.0, f"HOLD. {note}{base}", atr_v, indicators, exit_long=exit_long)


class RsiDipStrategy:
    """Buy short pullbacks inside an uptrend (RSI below 32 while above EMA200); sell when RSI recovers above 60."""

    name = "rsi_dip"

    def __init__(self, period: int = 14, buy_below: float = 32.0, sell_above: float = 60.0, filter_period: int = 200) -> None:
        self.period, self.buy_below, self.sell_above, self.filter_period = period, buy_below, sell_above, filter_period
        self.min_bars = filter_period + 10

    def evaluate(self, frame: pd.DataFrame) -> StrategyResult | None:
        if len(frame) < self.min_bars:
            return None
        close = frame["close"].astype("float64")
        rsi_v = _last(rsi(close, self.period))
        flt_v = _last(ema(close, self.filter_period))
        atr_v = _last(atr_series(frame, 14))
        price = _last(close)
        if not all(_finite(v) for v in (rsi_v, flt_v, atr_v, price)):
            return None
        indicators = {"close": price, "rsi": rsi_v, "ema_filter": flt_v, "atr": atr_v}
        base = f"RSI{self.period} {rsi_v:.1f}; EMA{self.filter_period} {flt_v:.8g}; close {price:.8g}"
        exit_long = rsi_v > self.sell_above
        if rsi_v < self.buy_below and price > flt_v:
            depth = (self.buy_below - rsi_v) / self.buy_below
            return StrategyResult(SignalAction.BUY, min(1.0, 0.65 + 1.5 * depth), f"BUY the dip: RSI is low while the long-term trend is up. {base}", atr_v, indicators)
        note = f"RSI above {self.sell_above:.0f}: exit longs. " if exit_long else ""
        return StrategyResult(SignalAction.HOLD, 0.0, f"HOLD. {note}{base}", atr_v, indicators, exit_long=exit_long)


class BollingerSqueezeStrategy:
    """Buy a breakout above the upper Bollinger band after an unusually quiet period; sell below the middle band."""

    name = "bb_squeeze"

    def __init__(self, period: int = 20, num_std: float = 2.0, lookback: int = 120, quiet_quantile: float = 0.25) -> None:
        self.period, self.num_std, self.lookback, self.quiet_quantile = period, num_std, lookback, quiet_quantile
        self.min_bars = lookback + period + 10

    def evaluate(self, frame: pd.DataFrame) -> StrategyResult | None:
        if len(frame) < self.min_bars:
            return None
        close = frame["close"].astype("float64")
        mid, upper, lower = bollinger_bands(close, self.period, self.num_std)
        width = (upper - lower) / mid
        recent = width.iloc[-self.lookback - 1 : -1].dropna()
        atr_v = _last(atr_series(frame, 14))
        price, up, md, w_prev = _last(close), _last(upper), _last(mid), _last(width, 2)
        if len(recent) < self.lookback // 2 or not all(_finite(v) for v in (price, up, md, w_prev, atr_v)):
            return None
        threshold = float(recent.quantile(self.quiet_quantile))
        indicators = {"close": price, "bb_upper": up, "bb_mid": md, "bb_width": _last(width), "squeeze_level": threshold, "atr": atr_v}
        base = f"close {price:.8g}; upper band {up:.8g}; middle {md:.8g}; band width {w_prev:.4f} (quiet below {threshold:.4f})"
        exit_long = price < md
        if w_prev <= threshold and price > up:
            return StrategyResult(SignalAction.BUY, 0.75, f"BUY: breakout after a quiet period. {base}", atr_v, indicators)
        note = "close below the middle band: exit longs. " if exit_long else ""
        return StrategyResult(SignalAction.HOLD, 0.0, f"HOLD. {note}{base}", atr_v, indicators, exit_long=exit_long)


class MacdTrendStrategy:
    """Buy when MACD crosses above its signal line while both the MACD and the long-term trend are positive."""

    name = "macd_trend"

    def __init__(self, filter_period: int = 200) -> None:
        self.filter_period = filter_period
        self.min_bars = filter_period + 10

    def evaluate(self, frame: pd.DataFrame) -> StrategyResult | None:
        if len(frame) < self.min_bars:
            return None
        close = frame["close"].astype("float64")
        line, signal_line, _ = macd(close)
        flt = ema(close, self.filter_period)
        atr_v = _last(atr_series(frame, 14))
        values = [_last(line), _last(line, 2), _last(signal_line), _last(signal_line, 2), _last(flt), _last(close), atr_v]
        if not all(_finite(v) for v in values):
            return None
        l_now, l_prev, s_now, s_prev, flt_v, price, atr_v = values
        indicators = {"close": price, "macd": l_now, "macd_signal": s_now, "ema_filter": flt_v, "atr": atr_v}
        base = f"MACD {l_now:.6g} vs signal {s_now:.6g}; EMA{self.filter_period} {flt_v:.8g}; close {price:.8g}"
        crossed_up = l_prev <= s_prev and l_now > s_now
        exit_long = l_now < s_now
        if crossed_up and l_now > 0 and price > flt_v:
            return StrategyResult(SignalAction.BUY, 0.75, f"BUY: MACD crossed above its signal in an uptrend. {base}", atr_v, indicators)
        note = "MACD below its signal: exit longs. " if exit_long else ""
        return StrategyResult(SignalAction.HOLD, 0.0, f"HOLD. {note}{base}", atr_v, indicators, exit_long=exit_long)


STRATEGY_NAMES = ("trend", "trend_fast", "ema_cross", "rsi_dip", "bb_squeeze", "macd_trend", "legacy")

# Plain-language names for the dashboard and reports.
STRATEGY_LABELS = {
    "trend": {"he": "מעקב מגמה", "en": "Trend following"},
    "trend_fast": {"he": "מעקב מגמה מהיר", "en": "Fast trend following"},
    "ema_cross": {"he": "חציית ממוצעים", "en": "Moving-average cross"},
    "rsi_dip": {"he": "קנייה בירידה זמנית", "en": "Buy the dip (RSI)"},
    "bb_squeeze": {"he": "פריצה אחרי שקט", "en": "Breakout after a quiet period"},
    "macd_trend": {"he": "מומנטום MACD", "en": "MACD momentum"},
    "legacy": {"he": "השיטה המקורית", "en": "Original mix"},
}

_BUILDERS = {
    "trend": TrendBreakoutStrategy,
    "trend_fast": TrendFastStrategy,
    "ema_cross": EmaCrossStrategy,
    "rsi_dip": RsiDipStrategy,
    "bb_squeeze": BollingerSqueezeStrategy,
    "macd_trend": MacdTrendStrategy,
    "legacy": LegacyScoreStrategy,
}


def build_strategy(name: str) -> Strategy:
    try:
        return _BUILDERS[name]()
    except KeyError:
        raise ValueError(f"Unknown strategy {name!r}") from None
