"""Indicator math against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents.market_data import IndicatorSnapshot, interpret, signal_from_frame
from core.indicators import atr, bollinger_bands, ema, macd, rsi, sma
from core.models import SignalAction


def test_sma_matches_a_plain_window_mean() -> None:
    result = sma(pd.Series([1, 2, 3, 4, 5, 6], dtype="float64"), 3)
    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert list(result.iloc[2:]) == [2, 3, 4, 5]


def test_ema_is_seeded_with_the_sma() -> None:
    result = ema(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    # alpha = 2 / 4 = 0.5, seed = mean(1, 2, 3) = 2
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_rsi_wilder_matches_manual_smoothing() -> None:
    # deltas: +1, +1, -1, -1 with period 2.
    # First RSI (index 2) is 100. Index 3 is 50. Index 4 is 25.
    result = rsi(pd.Series([10.0, 11.0, 12.0, 11.0, 10.0]), period=2)
    assert result.iloc[2] == pytest.approx(100.0)
    assert result.iloc[3] == pytest.approx(50.0)
    assert result.iloc[4] == pytest.approx(25.0)


def test_rsi_extremes() -> None:
    rising = pd.Series([float(value) for value in range(1, 20)])
    falling = pd.Series([float(value) for value in range(20, 0, -1)])
    assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(falling, 14).iloc[-1] == pytest.approx(0.0)


def test_atr_wilder_matches_manual_true_range() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 13.0, 12.0, 11.0],
            "low": [10.0, 11.0, 9.0, 10.0],
            "close": [10.0, 12.0, 11.0, 11.0],
        }
    )
    result = atr(frame, period=2)
    # TR after the seed bar: 3, 3, then next bar TR 1.
    # ATR at index 2 = 3. ATR at index 3 = (3 + 1) / 2 = 2.
    assert result.iloc[2] == pytest.approx(3.0)
    assert result.iloc[3] == pytest.approx(2.0)


def test_bollinger_uses_population_stddev() -> None:
    middle, upper, lower = bollinger_bands(pd.Series([1.0, 2.0, 3.0]), period=3, num_std=2.0)
    deviation = math.sqrt(2.0 / 3.0)
    assert middle.iloc[2] == pytest.approx(2.0)
    assert upper.iloc[2] == pytest.approx(2.0 + 2.0 * deviation)
    assert lower.iloc[2] == pytest.approx(2.0 - 2.0 * deviation)


def test_macd_histogram_is_line_minus_signal() -> None:
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 80)))
    line, signal, histogram = macd(close)
    valid = histogram.dropna().index
    assert len(valid) > 0
    assert np.allclose((line - signal).loc[valid], histogram.loc[valid], equal_nan=True)


def test_short_history_is_a_hold() -> None:
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.1, 1.1],
            "low": [0.9, 0.9],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    signal = signal_from_frame("BTC/USDT", frame)
    assert signal.action == SignalAction.HOLD
    assert signal.confidence == 0.0
    assert signal.price == pytest.approx(1.0)


def test_interpret_buy_sell_and_hold() -> None:
    buy = interpret(
        IndicatorSnapshot(
            price=100,
            rsi=28,
            macd=0.5,
            macd_signal=0.2,
            macd_hist=0.3,
            macd_hist_prev=0.1,
            sma_fast=98,
            sma_slow=90,
            ema_fast=99,
            ema_slow=95,
            bb_upper=110,
            bb_mid=100,
            bb_lower=90,
            atr=2,
        )
    )
    assert buy[0] == SignalAction.BUY
    assert buy[1] == pytest.approx(1.0)
    assert buy[3] == pytest.approx(0.85)

    sell = interpret(
        IndicatorSnapshot(
            price=90,
            rsi=75,
            macd=-0.4,
            macd_signal=-0.1,
            macd_hist=-0.4,
            macd_hist_prev=-0.1,
            sma_fast=100,
            sma_slow=110,
            ema_fast=90,
            ema_slow=100,
            bb_upper=120,
            bb_mid=100,
            bb_lower=80,
            atr=2,
        )
    )
    assert sell[0] == SignalAction.SELL
    assert sell[3] == pytest.approx(-0.85)

    hold = interpret(
        IndicatorSnapshot(
            price=100,
            rsi=50,
            macd=0,
            macd_signal=0,
            macd_hist=0,
            macd_hist_prev=0,
            sma_fast=100,
            sma_slow=100,
            ema_fast=100,
            ema_slow=100,
            bb_upper=110,
            bb_mid=100,
            bb_lower=90,
            atr=1,
        )
    )
    assert hold[0] == SignalAction.HOLD
    assert hold[1] == pytest.approx(0.0)


def test_collapsed_bollinger_band_does_not_count_as_a_break() -> None:
    action, _confidence, reasoning, raw = interpret(
        IndicatorSnapshot(
            price=100,
            rsi=50,
            macd=0,
            macd_signal=0,
            macd_hist=0,
            macd_hist_prev=0,
            sma_fast=100,
            sma_slow=100,
            ema_fast=100,
            ema_slow=100,
            bb_upper=100,
            bb_mid=100,
            bb_lower=100,
            atr=1,
        )
    )
    assert action == SignalAction.HOLD
    assert raw == pytest.approx(0.0)
    assert "collapsed" in reasoning
