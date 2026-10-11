"""Hand-rolled technical indicators.

The implementations are the classic definitions used by charting platforms:

* SMA is a simple rolling mean.
* EMA is seeded with the SMA of the first window, then
  ``EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}`` with
  ``alpha = 2 / (period + 1)``.
* RSI and ATR use Wilder's smoothing.
* Bollinger Bands use the population standard deviation (``ddof=0``).

No pandas-ta dependency: the math stays explicit and the install stays small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_FAST = 20
SMA_SLOW = 50
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
# Slow SMA plus enough bars for the MACD signal line to exist.
MIN_BARS = SMA_SLOW + MACD_SIGNAL


def sma(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    values = series.astype("float64")
    return values.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA seeded with an SMA. Leading NaNs are skipped so this works on MACD."""
    if period < 1:
        raise ValueError("period must be >= 1")
    values = series.to_numpy(dtype="float64", copy=False)
    out = np.full(values.shape, np.nan, dtype="float64")
    finite = np.isfinite(values)
    if not finite.any():
        return pd.Series(out, index=series.index)
    start = int(np.argmax(finite))
    segment = values[start:]
    if len(segment) < period or not np.isfinite(segment[:period]).all():
        return pd.Series(out, index=series.index)
    alpha = 2.0 / (period + 1.0)
    prev = float(segment[:period].mean())
    out[start + period - 1] = prev
    for offset in range(period, len(segment)):
        price = segment[offset]
        if not np.isfinite(price):
            continue
        prev = alpha * float(price) + (1.0 - alpha) * prev
        out[start + offset] = prev
    return pd.Series(out, index=series.index)


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI. Flat markets are 50, a pure rise is 100, a pure fall is 0."""
    if period < 1:
        raise ValueError("period must be >= 1")
    values = close.to_numpy(dtype="float64", copy=False)
    out = np.full(values.shape, np.nan, dtype="float64")
    if len(values) <= period:
        return pd.Series(out, index=close.index)
    deltas = np.diff(values)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for index in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + float(gains[index])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[index])) / period
        out[index + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return pd.Series(out, index=close.index)


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(macd_line, signal_line, histogram)``."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    histogram = line - signal_line
    return line, signal_line, histogram


def true_range(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].to_numpy(dtype="float64", copy=False)
    low = frame["low"].to_numpy(dtype="float64", copy=False)
    close = frame["close"].to_numpy(dtype="float64", copy=False)
    previous = np.empty_like(close)
    previous[0] = np.nan
    previous[1:] = close[:-1]
    parts = np.vstack([high - low, np.abs(high - previous), np.abs(low - previous)])
    # skipna: the first bar has no previous close, so high-low is the range.
    with np.errstate(all="ignore"):
        ranges = np.nanmax(parts, axis=0)
    return pd.Series(ranges, index=frame.index)


def atr(frame: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder ATR. The seed is the simple average of the first ``period`` true ranges."""
    if period < 1:
        raise ValueError("period must be >= 1")
    ranges = true_range(frame).to_numpy(dtype="float64", copy=False)
    out = np.full(ranges.shape, np.nan, dtype="float64")
    if len(ranges) <= period:
        return pd.Series(out, index=frame.index)
    seed = ranges[1 : period + 1]
    if len(seed) < period or not np.isfinite(seed).all():
        return pd.Series(out, index=frame.index)
    prev = float(seed.mean())
    out[period] = prev
    for index in range(period + 1, len(ranges)):
        prev = (prev * (period - 1) + float(ranges[index])) / period
        out[index] = prev
    return pd.Series(out, index=frame.index)


def bollinger_bands(
    close: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(middle, upper, lower)`` using population standard deviation."""
    values = close.astype("float64")
    middle = sma(values, period)
    deviation = values.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * deviation
    lower = middle - num_std * deviation
    return middle, upper, lower


def donchian(frame: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """Highest high and lowest low of the *previous* ``period`` bars.

    The current bar is excluded so ``close > upper`` is a genuine breakout
    instead of comparing a bar with itself.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    upper = high.rolling(window=period, min_periods=period).max().shift(1)
    lower = low.rolling(window=period, min_periods=period).min().shift(1)
    return upper, lower


def compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy ``frame`` and attach the columns the signal scorer reads."""
    out = frame.copy()
    close = out["close"].astype("float64")
    out["sma_fast"] = sma(close, SMA_FAST)
    out["sma_slow"] = sma(close, SMA_SLOW)
    out["ema_fast"] = ema(close, MACD_FAST)
    out["ema_slow"] = ema(close, MACD_SLOW)
    out["rsi"] = rsi(close, RSI_PERIOD)
    macd_line, macd_signal, macd_hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    middle, upper, lower = bollinger_bands(close, BB_PERIOD, BB_STD)
    out["bb_mid"] = middle
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["atr"] = atr(out, ATR_PERIOD)
    return out
