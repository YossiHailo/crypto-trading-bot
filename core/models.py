"""Signals, orders, and portfolio snapshots shared by every agent.

The market agent emits ``TechnicalSignal``. The sentiment agent emits
``SentimentSignal``. The risk agent is the only component allowed to turn
those into a ``RiskDecision``. The execution agent turns an executable
decision into an ``OrderResult`` and reports ``ClosedTrade`` events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SentimentBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class RiskAction(str, Enum):
    EXECUTE_BUY = "EXECUTE_BUY"
    EXECUTE_SELL = "EXECUTE_SELL"
    NO_ACTION = "NO_ACTION"


class CloseReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SIGNAL = "signal"
    MANUAL = "manual"
    EXTERNAL = "external"  # flat on the exchange, but not by an order the bot can identify


@dataclass(frozen=True)
class Bar:
    """High/low of one candle, used to detect stops touched between checks."""

    open_ms: int
    close_ms: int
    high: float
    low: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class TechnicalSignal:
    """Output of the market-data agent for one symbol."""

    symbol: str
    action: SignalAction
    confidence: float
    reasoning: str
    atr: float
    price: float
    timestamp: datetime
    indicators: dict[str, float] = field(default_factory=dict)
    # Exit-only hints. A trend strategy can say "close the long" without
    # saying "open a short", which a plain SELL would imply.
    exit_long: bool = False
    exit_short: bool = False
    strategy: str = ""
    # Open time (ms) of the last *closed* bar the signal was computed from.
    bar_time_ms: int | None = None
    # Recent bars (closed and forming) for intrabar stop checks.
    recent_bars: tuple[Bar, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.price < 0 or self.atr < 0:
            raise ValueError("price and atr must be non-negative")

    def wants_exit(self, side: str) -> bool:
        if side == "long":
            return self.action == SignalAction.SELL or self.exit_long
        return self.action == SignalAction.BUY or self.exit_short


@dataclass(frozen=True)
class SentimentSignal:
    """Output of the sentiment agent. ``score`` is -1 bearish to +1 bullish."""

    bias: SentimentBias
    score: float
    reasoning: str
    fear_greed: int | None = None
    headline_score: float | None = None
    timestamp: datetime = field(default_factory=utcnow)
    sources_used: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"sentiment score must be in [-1, 1], got {self.score}")


@dataclass(frozen=True)
class RiskDecision:
    """Output of the risk agent. Only EXECUTE_* decisions are sent to the exchange."""

    action: RiskAction
    symbol: str
    position_size: float
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float
    reasoning: str
    risk_amount: float = 0.0
    reward_risk: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.position_size < 0:
            raise ValueError("position_size must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class Position:
    """An open position tracked locally (paper fill or live order the bot placed)."""

    symbol: str
    side: str  # "long" or "short"
    size: float
    entry: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    open_fee: float
    margin_locked: float
    exchange_protected: bool = False
    order_id: str | None = None
    sl_order_id: str | None = None
    tp_order_id: str | None = None

    def notional(self, mark: float | None = None) -> float:
        return self.size * (mark if mark else self.entry)

    def unrealized(self, mark: float) -> float:
        if self.side == "long":
            return (mark - self.entry) * self.size
        return (self.entry - mark) * self.size


@dataclass
class PendingOrder:
    """A limit entry that has not filled yet."""

    id: str
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    limit_price: float
    stop_loss: float
    take_profit: float
    created_at: datetime
    client_order_id: str | None = None


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    side: str
    size: float
    entry: float
    exit_price: float
    pnl: float
    reason: CloseReason
    opened_at: datetime
    closed_at: datetime
    order_id: str | None = None


@dataclass(frozen=True)
class OrderResult:
    success: bool
    symbol: str
    side: str
    amount: float
    price: float
    status: str
    message: str
    order_id: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    opened: bool = False
    closed_trade: ClosedTrade | None = None


@dataclass
class MarkResult:
    """Stops, targets, and resting limits processed during a price update."""

    closed: list[ClosedTrade] = field(default_factory=list)
    opened: list[OrderResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PortfolioView:
    """Read-only account snapshot consumed by the risk agent."""

    cash: float
    equity: float
    day_start_equity: float
    realized_pnl_today: float
    day_start_date: str
    positions: dict[str, Position]
    cooldowns: dict[str, datetime]
    pending_symbols: set[str] = field(default_factory=set)
    quote_currency: str = "USDT"
    balance_synced: bool = True
    peak_equity: float = 0.0
    # Set by the max-drawdown kill switch or by the operator (dashboard).
    # Either one blocks new entries. Exits keep working.
    halted_reason: str | None = None
    paused: bool = False
    open_notional: float = 0.0

    @property
    def open_position_count(self) -> int:
        return len(self.positions)


@dataclass
class CycleReport:
    started_at: datetime
    finished_at: datetime | None = None
    signals: list[TechnicalSignal] = field(default_factory=list)
    sentiment: SentimentSignal | None = None
    decisions: list[RiskDecision] = field(default_factory=list)
    orders: list[OrderResult] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    daily_summary: str | None = None
    prices: dict[str, float] = field(default_factory=dict)


def neutral_sentiment(reason: str) -> SentimentSignal:
    """Used when every sentiment source fails. The bot keeps running."""
    return SentimentSignal(
        bias=SentimentBias.NEUTRAL,
        score=0.0,
        reasoning=f"Degrading to NEUTRAL. {reason}",
        fear_greed=None,
        headline_score=None,
        sources_used=(),
    )


def clamp_unit(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def clamp_score(value: float) -> float:
    return _clamp(value, -1.0, 1.0)
