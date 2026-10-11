"""Market data agent.

Fetches OHLCV through a candle source (CCXT by default), drops the candle
that is still forming, scores the closed bars with the configured strategy
(``agents/strategies.py``), and emits a ``TechnicalSignal``.

``interpret`` and ``snapshot_from_frame`` below are the original weighted
score, now wrapped by ``LegacyScoreStrategy``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from core.indicators import MIN_BARS, compute_indicators
from core.models import Bar, SignalAction, TechnicalSignal, utcnow
from core.timeframes import split_closed, timeframe_to_ms

logger = logging.getLogger("trading.market")

# Component weights. They sum to 1.0 so the raw score stays inside [-1, 1].
_RSI_WEIGHT = 0.30
_MACD_WEIGHT = 0.25
_TREND_WEIGHT = 0.20
_BB_WEIGHT = 0.15
_EMA_WEIGHT = 0.10
# Absolute raw score above this becomes BUY or SELL. Below it is HOLD.
_ACTION_THRESHOLD = 0.20
# A raw score of this magnitude is reported as confidence 1.0.
_CONFIDENCE_SCALE = 0.60


class CandleSource(Protocol):
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Return CCXT-style rows: ``[timestamp_ms, open, high, low, close, volume]``."""

    async def close(self) -> None:
        """Release the underlying client."""


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Last-bar indicator values passed to the scorer. Kept separate for tests."""

    price: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    sma_fast: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    atr: float


def ohlcv_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def snapshot_from_frame(frame: pd.DataFrame) -> IndicatorSnapshot | None:
    """Build a snapshot when the last bar has a full indicator set."""
    if len(frame) < MIN_BARS:
        return None
    enriched = compute_indicators(frame)
    last = enriched.iloc[-1]
    previous = enriched.iloc[-2]
    required = (
        "close",
        "rsi",
        "atr",
        "macd",
        "macd_signal",
        "macd_hist",
        "sma_fast",
        "sma_slow",
        "ema_fast",
        "ema_slow",
        "bb_upper",
        "bb_mid",
        "bb_lower",
    )
    if any(not _finite(last[column]) for column in required):
        return None
    previous_hist = float(previous["macd_hist"]) if _finite(previous["macd_hist"]) else float(last["macd_hist"])
    return IndicatorSnapshot(
        price=float(last["close"]),
        rsi=float(last["rsi"]),
        macd=float(last["macd"]),
        macd_signal=float(last["macd_signal"]),
        macd_hist=float(last["macd_hist"]),
        macd_hist_prev=previous_hist,
        sma_fast=float(last["sma_fast"]),
        sma_slow=float(last["sma_slow"]),
        ema_fast=float(last["ema_fast"]),
        ema_slow=float(last["ema_slow"]),
        bb_upper=float(last["bb_upper"]),
        bb_mid=float(last["bb_mid"]),
        bb_lower=float(last["bb_lower"]),
        atr=max(0.0, float(last["atr"])),
    )


def interpret(snapshot: IndicatorSnapshot) -> tuple[SignalAction, float, str, float]:
    """Score one snapshot.

    Returns ``(action, confidence, reasoning, raw_score)``. Confidence is
    ``min(1, abs(raw) / 0.60)`` so a moderate alignment clears a 0.55 gate
    and a perfect alignment saturates at 1.
    """
    notes: list[tuple[str, float, str]] = []

    rsi_value = snapshot.rsi
    if rsi_value <= 30:
        notes.append(("rsi", _RSI_WEIGHT, f"RSI {rsi_value:.1f} oversold"))
    elif rsi_value <= 40:
        notes.append(("rsi", _RSI_WEIGHT / 2, f"RSI {rsi_value:.1f} leaning oversold"))
    elif rsi_value >= 70:
        notes.append(("rsi", -_RSI_WEIGHT, f"RSI {rsi_value:.1f} overbought"))
    elif rsi_value >= 60:
        notes.append(("rsi", -_RSI_WEIGHT / 2, f"RSI {rsi_value:.1f} leaning overbought"))
    else:
        notes.append(("rsi", 0.0, f"RSI {rsi_value:.1f} neutral"))

    hist = snapshot.macd_hist
    prev = snapshot.macd_hist_prev
    if hist > 0 and hist >= prev:
        notes.append(("macd", _MACD_WEIGHT, "MACD histogram positive and rising"))
    elif hist > 0:
        notes.append(("macd", _MACD_WEIGHT * 0.4, "MACD histogram positive but fading"))
    elif hist < 0 and hist <= prev:
        notes.append(("macd", -_MACD_WEIGHT, "MACD histogram negative and falling"))
    elif hist < 0:
        notes.append(("macd", -_MACD_WEIGHT * 0.4, "MACD histogram negative but contracting"))
    else:
        notes.append(("macd", 0.0, "MACD histogram flat"))

    if snapshot.price > snapshot.sma_fast > snapshot.sma_slow:
        notes.append(("trend", _TREND_WEIGHT, "price above SMA20 and SMA20 above SMA50"))
    elif snapshot.price < snapshot.sma_fast < snapshot.sma_slow:
        notes.append(("trend", -_TREND_WEIGHT, "price below SMA20 and SMA20 below SMA50"))
    else:
        notes.append(("trend", 0.0, "SMA trend mixed"))

    bandwidth = snapshot.bb_upper - snapshot.bb_lower
    if bandwidth <= 1e-12:
        notes.append(("bollinger", 0.0, "Bollinger bands collapsed"))
    elif snapshot.price <= snapshot.bb_lower:
        notes.append(("bollinger", _BB_WEIGHT, "price at or below the lower band"))
    elif snapshot.price >= snapshot.bb_upper:
        notes.append(("bollinger", -_BB_WEIGHT, "price at or above the upper band"))
    else:
        notes.append(("bollinger", 0.0, "price inside the bands"))

    if snapshot.ema_fast > snapshot.ema_slow:
        notes.append(("ema", _EMA_WEIGHT, "EMA12 above EMA26"))
    elif snapshot.ema_fast < snapshot.ema_slow:
        notes.append(("ema", -_EMA_WEIGHT, "EMA12 below EMA26"))
    else:
        notes.append(("ema", 0.0, "EMA12 and EMA26 are equal"))

    raw = max(-1.0, min(1.0, sum(weight for _, weight, _ in notes)))
    if raw > _ACTION_THRESHOLD:
        action = SignalAction.BUY
    elif raw < -_ACTION_THRESHOLD:
        action = SignalAction.SELL
    else:
        action = SignalAction.HOLD
    confidence = min(1.0, abs(raw) / _CONFIDENCE_SCALE)
    detail = "; ".join(f"{name}: {text} ({weight:+.2f})" for name, weight, text in notes)
    reasoning = f"score {raw:+.2f}. {detail}"
    return action, confidence, reasoning, raw


def signal_from_frame(
    symbol: str,
    frame: pd.DataFrame,
    *,
    now: datetime | None = None,
    strategy: object | None = None,
    price: float | None = None,
    recent_bars: tuple[Bar, ...] = (),
) -> TechnicalSignal:
    """Score a frame of closed candles. Short or incomplete history becomes HOLD.

    ``price`` is the latest traded price (the forming candle's close). It
    defaults to the last closed close.
    """
    from agents.strategies import LegacyScoreStrategy

    strategy = strategy or LegacyScoreStrategy()
    timestamp = now or utcnow()
    last_close = 0.0
    if len(frame) and _finite(frame["close"].iloc[-1]):
        last_close = float(frame["close"].iloc[-1])
    current = price if price is not None and price > 0 else last_close
    bar_time = int(frame["timestamp"].iloc[-1]) if len(frame) and "timestamp" in frame and _finite(frame["timestamp"].iloc[-1]) else None
    result = strategy.evaluate(frame) if len(frame) else None  # type: ignore[attr-defined]
    name = getattr(strategy, "name", "")
    if result is None:
        need = getattr(strategy, "min_bars", MIN_BARS)
        return TechnicalSignal(
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            reasoning=f"Insufficient or incomplete candles ({len(frame)} rows, need {need}).",
            atr=0.0,
            price=current,
            timestamp=timestamp,
            strategy=name,
            bar_time_ms=bar_time,
            recent_bars=recent_bars,
        )
    indicators = dict(result.indicators)
    if price is not None and price > 0 and last_close > 0:
        indicators["last_closed"] = last_close
    return TechnicalSignal(
        symbol=symbol,
        action=result.action,
        confidence=result.confidence,
        reasoning=result.reasoning,
        atr=result.atr,
        price=current,
        timestamp=timestamp,
        indicators=indicators,
        exit_long=result.exit_long,
        exit_short=result.exit_short,
        strategy=name,
        bar_time_ms=bar_time,
        recent_bars=recent_bars,
    )


class CcxtCandleSource:
    """Public OHLCV adapter around an async CCXT exchange."""

    def __init__(self, exchange: object, timeout_seconds: float) -> None:
        self.exchange = exchange
        self.timeout_seconds = timeout_seconds

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        fetch = self.exchange.fetch_ohlcv  # type: ignore[attr-defined]
        rows = await asyncio.wait_for(
            fetch(symbol, timeframe=timeframe, limit=limit),
            timeout=self.timeout_seconds,
        )
        return rows

    async def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


class FallbackCandleSource:
    """Try each candle source in order. Useful when one exchange blocks a region."""

    def __init__(self, sources: list[CandleSource]) -> None:
        if not sources:
            raise ValueError("at least one source is required")
        self.sources = sources

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        errors = []
        for index, source in enumerate(self.sources):
            try:
                rows = await source.fetch_ohlcv(symbol, timeframe, limit)
                if rows:
                    if index:
                        logger.info("%s candles from fallback source #%d", symbol, index)
                    return rows
                errors.append(f"source {index}: no candles")
            except Exception as exc:
                errors.append(f"source {index}: {' '.join(str(exc).split())[:200]}")
        raise RuntimeError("all candle sources failed: " + " | ".join(errors))

    async def close(self) -> None:
        for source in self.sources:
            try:
                await source.close()
            except Exception:
                logger.debug("close failed", exc_info=True)


def _bars(rows: list[list[float]], timeframe: str, count: int = 3) -> tuple[Bar, ...]:
    step = timeframe_to_ms(timeframe)
    out: list[Bar] = []
    for row in rows[-count:]:
        try:
            out.append(Bar(open_ms=int(row[0]), close_ms=int(row[0]) + step, high=float(row[2]), low=float(row[3])))
        except (TypeError, ValueError, IndexError):
            continue
    return tuple(out)


class MarketDataAgent:
    """Fetches candles and scores the latest *closed* bar."""

    def __init__(
        self,
        source: CandleSource,
        timeframe: str,
        limit: int,
        *,
        strategy: object | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        from agents.strategies import LegacyScoreStrategy

        self.source = source
        self.timeframe = timeframe
        self.limit = limit
        self.strategy = strategy or LegacyScoreStrategy()
        self._clock = clock or utcnow

    async def analyze(self, symbol: str) -> TechnicalSignal:
        now = self._clock()
        # One extra row because the forming candle is dropped before scoring.
        rows = await self.source.fetch_ohlcv(symbol, self.timeframe, self.limit + 1)
        if not rows:
            logger.warning("No candles returned for %s", symbol)
            return TechnicalSignal(
                symbol=symbol,
                action=SignalAction.HOLD,
                confidence=0.0,
                reasoning="Exchange returned no candles.",
                atr=0.0,
                price=0.0,
                timestamp=now,
                strategy=getattr(self.strategy, "name", ""),
            )
        closed, forming = split_closed(rows, self.timeframe, int(now.timestamp() * 1000))
        price = None
        if forming is not None and _finite(forming[4]):
            price = float(forming[4])
        signal = signal_from_frame(
            symbol,
            ohlcv_to_frame(closed),
            now=now,
            strategy=self.strategy,
            price=price,
            recent_bars=_bars(rows, self.timeframe),
        )
        logger.info(
            "%s %s confidence=%.2f price=%s atr=%s",
            symbol,
            signal.action.value,
            signal.confidence,
            f"{signal.price:.8g}",
            f"{signal.atr:.8g}",
        )
        logger.debug("%s reasoning: %s", symbol, signal.reasoning)
        return signal

    async def close(self) -> None:
        await self.source.close()
