"""Bar-by-bar backtest that reuses the live strategy and risk code.

Timing rules (no look-ahead):

* The signal for bar ``i`` is computed from closed bars only, on a sliding
  window of ``OHLCV_LIMIT`` bars, exactly like the live loop.
* Orders decided at the close of bar ``i`` fill at the **open of bar i+1**,
  plus slippage, plus the fee.
* Stops and targets are checked against each later bar's high and low. A gap
  through the stop fills at the open. When one bar touches both the stop and
  the target, the stop is assumed to fill first (pessimistic).

Sizing, caps, confidence floor, daily loss breaker and cooldowns come from
``RiskManager`` with the same settings the bot uses. Sentiment is NEUTRAL
unless a Fear & Greed history is supplied (headlines have no history).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from agents.risk import RiskManager
from agents.sentiment import build_sentiment
from config.settings import Settings
from core.indicators import atr as atr_series
from core.models import (
    PortfolioView,
    Position,
    RiskAction,
    SentimentSignal,
    TechnicalSignal,
    neutral_sentiment,
)
from core.timeframes import bars_per_year, timeframe_to_ms


@dataclass
class Trade:
    side: str
    entry_time: datetime
    exit_time: datetime
    entry: float
    exit: float
    size: float
    pnl: float
    fees: float
    risk: float  # money at risk at entry (stop distance x size)
    reason: str
    bars_held: int

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk if self.risk > 0 else 0.0


@dataclass
class Metrics:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    avg_win: float
    avg_loss: float
    avg_bars_held: float
    exposure: float
    fees: float


@dataclass
class BacktestResult:
    name: str
    times: list[datetime]
    equity: np.ndarray
    trades: list[Trade]
    metrics: Metrics
    in_market: np.ndarray
    kill_switch_at: datetime | None = None
    notes: list[str] = field(default_factory=list)


def rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = frame["timestamp"].astype("int64")
    return frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ---------------------------------------------------------------- metrics


def compute_metrics(
    equity: np.ndarray,
    times: list[datetime],
    trades: list[Trade],
    in_market: np.ndarray,
    timeframe: str,
) -> Metrics:
    start, end = float(equity[0]), float(equity[-1])
    total = end / start - 1 if start > 0 else 0.0
    years = max((times[-1] - times[0]).total_seconds() / (365 * 86400), 1e-9)
    cagr = (end / start) ** (1 / years) - 1 if start > 0 and end > 0 else -1.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / np.where(peaks > 0, peaks, 1)
    max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0.0])
    annual = math.sqrt(bars_per_year(timeframe))
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * annual) if std > 0 else 0.0
    downside = returns[returns < 0]
    dstd = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0
    sortino = float(returns.mean() / dstd * annual) if dstd > 0 else 0.0
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    gross_loss = -sum(losses)
    return Metrics(
        total_return=total,
        cagr=cagr,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=(cagr / max_dd) if max_dd > 0 else 0.0,
        trades=len(trades),
        win_rate=(len(wins) / len(trades)) if trades else 0.0,
        profit_factor=(sum(wins) / gross_loss) if gross_loss > 0 else (float("inf") if wins else 0.0),
        expectancy_r=float(np.mean([t.r_multiple for t in trades])) if trades else 0.0,
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        avg_bars_held=float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        exposure=float(in_market.mean()) if len(in_market) else 0.0,
        fees=sum(t.fees for t in trades),
    )


# ----------------------------------------------------------- account model


@dataclass
class _Book:
    cash: float
    fee: float
    slip: float
    position: Position | None = None
    entry_index: int = 0
    entry_fees: float = 0.0
    entry_risk: float = 0.0

    def equity(self, price: float) -> float:
        if self.position is None:
            return self.cash
        p = self.position
        if p.side == "long":
            return self.cash + p.size * price
        return self.cash + p.margin_locked + (p.entry - price) * p.size

    def open(self, side: str, size: float, raw_price: float, stop: float, target: float, when: datetime, index: int) -> bool:
        fill = raw_price * (1 + self.slip) if side == "long" else raw_price * (1 - self.slip)
        if side == "long":
            size = min(size, self.cash / (fill * (1 + self.fee)))
        else:
            size = min(size, self.cash / (fill * (1 + self.fee)))
        if size <= 0:
            return False
        notional = size * fill
        fee = notional * self.fee
        margin = notional if side == "short" else 0.0
        self.cash -= (notional + fee) if side == "long" else (margin + fee)
        self.position = Position(
            symbol="BT",
            side=side,
            size=size,
            entry=fill,
            stop_loss=stop,
            take_profit=target,
            opened_at=when,
            open_fee=fee,
            margin_locked=margin,
            exchange_protected=True,
        )
        self.entry_index = index
        self.entry_fees = fee
        self.entry_risk = abs(fill - stop) * size
        return True

    def close(self, raw_price: float, when: datetime, index: int, reason: str, *, slip: bool = True) -> Trade:
        p = self.position
        assert p is not None
        if slip:
            fill = raw_price * (1 - self.slip) if p.side == "long" else raw_price * (1 + self.slip)
        else:
            fill = raw_price
        fee = p.size * fill * self.fee
        if p.side == "long":
            self.cash += p.size * fill - fee
            gross = (fill - p.entry) * p.size
        else:
            gross = (p.entry - fill) * p.size
            self.cash += p.margin_locked + gross - fee
        trade = Trade(
            side=p.side,
            entry_time=p.opened_at,
            exit_time=when,
            entry=p.entry,
            exit=fill,
            size=p.size,
            pnl=gross - p.open_fee - fee,
            fees=p.open_fee + fee,
            risk=self.entry_risk,
            reason=reason,
            bars_held=index - self.entry_index,
        )
        self.position = None
        return trade


def _check_bar_exit(book: _Book, o: float, h: float, low: float) -> tuple[float, str, bool] | None:
    """Return (price, reason, apply_slippage) if the stop or target was hit inside this bar."""
    p = book.position
    if p is None:
        return None
    if p.side == "long":
        if o <= p.stop_loss:
            return o, "stop_loss", True  # gapped through
        if low <= p.stop_loss:
            return p.stop_loss, "stop_loss", True
        if o >= p.take_profit:
            return o, "take_profit", False
        if h >= p.take_profit:
            return p.take_profit, "take_profit", False
    else:
        if o >= p.stop_loss:
            return o, "stop_loss", True
        if h >= p.stop_loss:
            return p.stop_loss, "stop_loss", True
        if o <= p.take_profit:
            return o, "take_profit", False
        if low <= p.take_profit:
            return p.take_profit, "take_profit", False
    return None


# -------------------------------------------------------------- strategy run


def _sentiment_for(day: date, fear_greed: dict[date, int] | None) -> SentimentSignal:
    if not fear_greed:
        return neutral_sentiment("Backtest: no sentiment history.")
    # Yesterday's published value, so the bar never sees today's number early.
    value = fear_greed.get(day - timedelta(days=1))
    if value is None:
        return neutral_sentiment("Backtest: Fear & Greed missing for this day.")
    return build_sentiment(fear_greed=value, fear_greed_label_text=None, headlines=None, errors=[])


def run_strategy(
    frame: pd.DataFrame,
    settings: Settings,
    strategy: object,
    *,
    symbol: str = "BT/USDT",
    fear_greed: dict[date, int] | None = None,
    start_index: int | None = None,
    name: str | None = None,
) -> BacktestResult:
    """Replay ``frame`` (closed candles, oldest first) through strategy + risk."""
    window = settings.ohlcv_limit
    min_bars = getattr(strategy, "min_bars", 60)
    first = max(window - 1, min_bars) if start_index is None else start_index
    n = len(frame)
    if n - first < 3:
        raise ValueError(f"Not enough candles: have {n}, need more than {first + 3}.")
    risk = RiskManager(settings)
    book = _Book(cash=settings.starting_balance, fee=settings.fee_rate, slip=settings.slippage_pct)
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    stamps = frame["timestamp"].to_numpy("int64")
    step = timeframe_to_ms(settings.timeframe)
    times: list[datetime] = []
    equity: list[float] = []
    in_market: list[bool] = []
    trades: list[Trade] = []
    pending: tuple[str, object] | None = None  # ("open", decision) or ("close", None)
    cooldowns: dict[str, datetime] = {}
    day_start_equity = book.cash
    current_day: date | None = None
    peak = book.cash
    kill_at: datetime | None = None

    for i in range(first, n):
        bar_open_time = _ts(int(stamps[i]))
        bar_close_time = _ts(int(stamps[i]) + step)
        # 1. Orders decided at the previous close fill at this open.
        if pending is not None:
            kind, payload = pending
            pending = None
            if kind == "close" and book.position is not None:
                trades.append(book.close(opens[i], bar_open_time, i, "signal"))
            elif kind == "open" and book.position is None:
                decision = payload
                side = "long" if decision.action == RiskAction.EXECUTE_BUY else "short"
                delta = opens[i] - decision.entry
                book.open(
                    side,
                    decision.position_size,
                    opens[i],
                    decision.stop_loss + delta,
                    decision.take_profit + delta,
                    bar_open_time,
                    i,
                )
        # 2. Stops and targets inside this bar.
        hit = _check_bar_exit(book, opens[i], highs[i], lows[i])
        if hit is not None:
            price, reason, slip = hit
            trade = book.close(price, bar_close_time, i, reason, slip=slip)
            trades.append(trade)
            if reason == "stop_loss":
                cooldowns[symbol] = bar_close_time + timedelta(minutes=settings.cooldown_minutes)
        # 3. Mark to the close, update the day and the peak.
        value = book.equity(closes[i])
        times.append(bar_close_time)
        equity.append(value)
        in_market.append(book.position is not None)
        if current_day != bar_close_time.date():
            current_day = bar_close_time.date()
            day_start_equity = value
        peak = max(peak, value)
        if kill_at is None and peak > 0 and (peak - value) / peak >= settings.max_drawdown_pct:
            kill_at = bar_close_time
        # 4. Decide at the close for the next open.
        if i == n - 1:
            break
        window_frame = frame.iloc[max(0, i - window + 1) : i + 1]
        result = strategy.evaluate(window_frame)  # type: ignore[attr-defined]
        if result is None:
            continue
        signal = TechnicalSignal(
            symbol=symbol,
            action=result.action,
            confidence=result.confidence,
            reasoning=result.reasoning,
            atr=result.atr,
            price=float(closes[i]),
            timestamp=bar_close_time,
            exit_long=result.exit_long,
            exit_short=result.exit_short,
            strategy=getattr(strategy, "name", ""),
        )
        positions = {}
        if book.position is not None:
            positions[symbol] = Position(**{**book.position.__dict__, "symbol": symbol})
        open_notional = book.position.size * closes[i] if book.position is not None else 0.0
        view = PortfolioView(
            cash=book.cash,
            equity=value,
            day_start_equity=day_start_equity,
            realized_pnl_today=0.0,
            day_start_date=str(current_day),
            positions=positions,
            cooldowns={k: v for k, v in cooldowns.items() if v > bar_close_time},
            quote_currency=settings.quote_currency,
            open_notional=open_notional,
        )
        decision = risk.decide(signal, _sentiment_for(bar_close_time.date(), fear_greed), view, now=bar_close_time)
        if decision.action == RiskAction.NO_ACTION:
            continue
        if book.position is not None:
            pending = ("close", None)
        else:
            pending = ("open", decision)

    if book.position is not None:
        trades.append(book.close(closes[-1], times[-1], n - 1, "end_of_data"))
        equity[-1] = book.cash
    eq = np.array(equity, dtype=float)
    market = np.array(in_market, dtype=bool)
    result = BacktestResult(
        name=name or getattr(strategy, "name", "strategy"),
        times=times,
        equity=eq,
        trades=trades,
        metrics=compute_metrics(eq, times, trades, market, settings.timeframe),
        in_market=market,
        kill_switch_at=kill_at,
    )
    if result.metrics.trades < 30:
        result.notes.append(
            f"Only {result.metrics.trades} trades. Too few to separate skill from luck; use a longer history or more symbols."
        )
    return result


# --------------------------------------------------------------- benchmarks


def buy_and_hold(frame: pd.DataFrame, settings: Settings, start_index: int) -> BacktestResult:
    book = _Book(cash=settings.starting_balance, fee=settings.fee_rate, slip=settings.slippage_pct)
    opens = frame["open"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    stamps = frame["timestamp"].to_numpy("int64")
    step = timeframe_to_ms(settings.timeframe)
    start = _ts(int(stamps[start_index]))
    book.open("long", float("inf"), opens[start_index], 0.0, float("inf"), start, start_index)
    times, equity = [], []
    for i in range(start_index, len(frame)):
        times.append(_ts(int(stamps[i]) + step))
        equity.append(book.equity(closes[i]))
    trade = book.close(closes[-1], times[-1], len(frame) - 1, "end_of_data")
    equity[-1] = book.cash
    eq = np.array(equity, dtype=float)
    market = np.ones(len(eq), dtype=bool)
    return BacktestResult("buy_and_hold", times, eq, [trade], compute_metrics(eq, times, [trade], market, settings.timeframe), market)


@dataclass
class RandomBenchmark:
    runs: int
    total_returns: np.ndarray
    sharpes: np.ndarray
    max_drawdowns: np.ndarray
    band_low: np.ndarray
    band_mid: np.ndarray
    band_high: np.ndarray

    def percentile_of(self, value: float, metric: str = "total_returns") -> float:
        data = getattr(self, metric)
        if len(data) == 0:
            return 0.0
        return float((data < value).mean() * 100)


def random_entries(
    frame: pd.DataFrame,
    settings: Settings,
    start_index: int,
    reference: BacktestResult,
    *,
    runs: int = 200,
    seed: int = 7,
) -> RandomBenchmark:
    """Same sizing, stops, targets, fees and holding time; entries chosen at random.

    If the strategy cannot beat this, its entries add nothing over chance and
    any profit came from the risk rules or from the market's drift.
    """
    rng = np.random.default_rng(seed)
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    atr_full = atr_series(frame).to_numpy(float)
    n = len(frame)
    span = n - start_index
    trade_count = max(1, reference.metrics.trades)
    flat_bars = max(1, int(span * (1 - reference.metrics.exposure)))
    p_enter = min(1.0, trade_count / flat_bars)
    holds = [t.bars_held for t in reference.trades] or [10]
    allow_short = settings.allow_short
    curves = np.empty((runs, span), dtype=float)
    totals, sharpes, dds = [], [], []
    annual = math.sqrt(bars_per_year(settings.timeframe))
    for run in range(runs):
        book = _Book(cash=settings.starting_balance, fee=settings.fee_rate, slip=settings.slippage_pct)
        exit_at = -1
        want_open = False
        side = "long"
        for k, i in enumerate(range(start_index, n)):
            if want_open and book.position is None:
                want_open = False
                stop_distance = (atr_full[i - 1] * settings.atr_multiplier) if settings.use_atr_stops and atr_full[i - 1] > 0 else closes[i - 1] * settings.fixed_stop_pct
                if stop_distance > 0 and math.isfinite(stop_distance):
                    equity_now = book.equity(closes[i - 1])
                    size = equity_now * settings.risk_per_trade / stop_distance
                    size = min(size, equity_now * settings.max_notional_pct / opens[i])
                    if side == "long":
                        stop, target = opens[i] - stop_distance, opens[i] + stop_distance * settings.reward_risk_ratio
                    else:
                        stop, target = opens[i] + stop_distance, opens[i] - stop_distance * settings.reward_risk_ratio
                    if stop > 0 and target > 0 and book.open(side, size, opens[i], stop, target, datetime.min, i):
                        exit_at = i + int(rng.choice(holds))
            hit = _check_bar_exit(book, opens[i], highs[i], lows[i])
            if hit is not None:
                book.close(hit[0], datetime.min, i, hit[1], slip=hit[2])
            elif book.position is not None and i >= exit_at:
                book.close(closes[i], datetime.min, i, "time")
            curves[run, k] = book.equity(closes[i])
            if book.position is None and not want_open and rng.random() < p_enter:
                want_open = True
                side = "short" if allow_short and rng.random() < 0.5 else "long"
        eq = curves[run]
        totals.append(eq[-1] / eq[0] - 1)
        rets = np.diff(eq) / eq[:-1]
        sd = rets.std(ddof=1) if len(rets) > 1 else 0
        sharpes.append(float(rets.mean() / sd * annual) if sd > 0 else 0.0)
        peaks = np.maximum.accumulate(eq)
        dds.append(float(((peaks - eq) / peaks).max()))
    return RandomBenchmark(
        runs=runs,
        total_returns=np.array(totals),
        sharpes=np.array(sharpes),
        max_drawdowns=np.array(dds),
        band_low=np.percentile(curves, 5, axis=0),
        band_mid=np.percentile(curves, 50, axis=0),
        band_high=np.percentile(curves, 95, axis=0),
    )


# ------------------------------------------------------------- walk-forward


@dataclass
class PeriodRow:
    start: datetime
    end: datetime
    strategy_return: float
    strategy_max_dd: float
    hold_return: float
    hold_max_dd: float
    trades: int


def period_breakdown(result: BacktestResult, hold: BacktestResult, folds: int) -> list[PeriodRow]:
    """Split the test into equal consecutive periods with the *same* fixed parameters.

    Nothing is fitted, so every period is out-of-sample. A strategy that only
    works in one period (one bull run) is fragile.
    """
    n = min(len(result.equity), len(hold.equity))
    if folds < 1 or n < folds * 2:
        return []
    edges = np.linspace(0, n - 1, folds + 1).astype(int)

    def ret(eq: np.ndarray, a: int, b: int) -> float:
        return float(eq[b] / eq[a] - 1)

    def mdd(eq: np.ndarray, a: int, b: int) -> float:
        seg = eq[a : b + 1]
        peaks = np.maximum.accumulate(seg)
        return float(((peaks - seg) / peaks).max())

    rows = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        start, end = result.times[a], result.times[b]
        rows.append(
            PeriodRow(
                start=start,
                end=end,
                strategy_return=ret(result.equity, a, b),
                strategy_max_dd=mdd(result.equity, a, b),
                hold_return=ret(hold.equity, a, b),
                hold_max_dd=mdd(hold.equity, a, b),
                trades=sum(1 for t in result.trades if start <= t.exit_time <= end),
            )
        )
    return rows


def verdict(result: BacktestResult, hold: BacktestResult, randoms: RandomBenchmark | None, periods: list[PeriodRow]) -> list[str]:
    """Plain-language conclusions. Deliberately conservative."""
    m, h = result.metrics, hold.metrics
    lines = []
    if m.trades < 30:
        lines.append(f"INCONCLUSIVE: only {m.trades} trades.")
    if randoms is not None:
        pct = randoms.percentile_of(m.total_return)
        lines.append(f"Return beats {pct:.0f}% of {randoms.runs} random-entry runs with the same risk rules.")
        if pct < 90:
            lines.append("NO EVIDENCE OF EDGE: the entries do not clearly beat chance (need roughly >= 95th percentile).")
        elif pct >= 95:
            lines.append("Entries beat random at the 95th percentile here. Promising, but confirm on other symbols and periods; one test can still be luck.")
    if m.sharpe < h.sharpe:
        lines.append(f"Risk-adjusted, buy-and-hold did better (Sharpe {h.sharpe:.2f} vs {m.sharpe:.2f}).")
    else:
        lines.append(f"Higher Sharpe than buy-and-hold ({m.sharpe:.2f} vs {h.sharpe:.2f}) with max drawdown {m.max_drawdown:.0%} vs {h.max_drawdown:.0%}.")
    if periods:
        positive = sum(1 for p in periods if p.strategy_return > 0)
        lines.append(f"Profitable in {positive} of {len(periods)} periods.")
        if positive <= len(periods) // 2:
            lines.append("Results depend on a few periods; treat them as regime luck until proven otherwise.")
    if result.kill_switch_at is not None:
        lines.append(f"The {result.metrics.max_drawdown:.0%} drawdown would have tripped the live kill switch on {result.kill_switch_at:%Y-%m-%d}.")
    return lines
