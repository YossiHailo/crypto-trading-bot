"""Shared builders. Bot environment variables are cleared so tests stay local."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from config.settings import Settings
from core.models import (
    PortfolioView,
    Position,
    SentimentBias,
    SentimentSignal,
    SignalAction,
    TechnicalSignal,
)

_ENV_KEYS = [
    "PAPER_TRADING",
    "LIVE_TRADING_CONFIRM",
    "EXCHANGE_ID",
    "EXCHANGE_API_KEY",
    "EXCHANGE_API_SECRET",
    "EXCHANGE_PASSWORD",
    "EXCHANGE_SANDBOX",
    "MARKET_TYPE",
    "SYMBOLS",
    "TIMEFRAME",
    "LOOP_INTERVAL_MINUTES",
    "OHLCV_LIMIT",
    "RISK_PER_TRADE",
    "REWARD_RISK_RATIO",
    "MIN_REWARD_RISK",
    "MIN_CONFIDENCE",
    "MAX_OPEN_POSITIONS",
    "DAILY_MAX_LOSS_PCT",
    "COOLDOWN_MINUTES",
    "ATR_MULTIPLIER",
    "USE_ATR_STOPS",
    "FIXED_STOP_PCT",
    "MAX_NOTIONAL_PCT",
    "ALLOW_SHORT",
    "ORDER_TYPE",
    "FEE_RATE",
    "CASH_UTILIZATION",
    "MIN_NOTIONAL",
    "STARTING_BALANCE",
    "QUOTE_CURRENCY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "CRYPTOPANIC_TOKEN",
    "FEAR_GREED_URL",
    "RSS_FEEDS",
    "LOG_LEVEL",
    "DATA_DIR",
    "LOG_DIR",
    "REQUEST_TIMEOUT_SECONDS",
    "RETRY_ATTEMPTS",
    "RETRY_BASE_DELAY",
    "ORDER_POLL_ATTEMPTS",
    "ORDER_POLL_DELAY",
    "STRATEGY",
    "LEVERAGE",
    "MAX_DRAWDOWN_PCT",
    "MAX_TOTAL_EXPOSURE_PCT",
    "SLIPPAGE_PCT",
    "PENDING_ORDER_TTL_MINUTES",
    "DASHBOARD_ENABLED",
    "DASHBOARD_HOST",
    "DASHBOARD_PORT",
    "DASHBOARD_TOKEN",
    "CLIENT_ORDER_PREFIX",
    "DECISIONS_RETENTION_DAYS",
]


@pytest.fixture(autouse=True)
def _clear_bot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def technical(
    *,
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.9,
    price: float = 100.0,
    atr: float = 2.0,
    symbol: str = "BTC/USDT",
    reasoning: str = "test signal",
) -> TechnicalSignal:
    return TechnicalSignal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        atr=atr,
        price=price,
        timestamp=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
    )


def sentiment(
    score: float = 0.0,
    bias: SentimentBias | None = None,
) -> SentimentSignal:
    if bias is None:
        if score > 0.15:
            bias = SentimentBias.BULLISH
        elif score < -0.15:
            bias = SentimentBias.BEARISH
        else:
            bias = SentimentBias.NEUTRAL
    return SentimentSignal(bias=bias, score=score, reasoning="test sentiment")


def portfolio(**overrides: object) -> PortfolioView:
    data: dict[str, object] = {
        "cash": 10_000.0,
        "equity": 10_000.0,
        "day_start_equity": 10_000.0,
        "realized_pnl_today": 0.0,
        "day_start_date": "2026-10-06",
        "positions": {},
        "cooldowns": {},
    }
    data.update(overrides)
    return PortfolioView(**data)  # type: ignore[arg-type]


def long_position(symbol: str = "BTC/USDT", size: float = 1.0) -> Position:
    return Position(
        symbol=symbol,
        side="long",
        size=size,
        entry=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        opened_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        open_fee=0.1,
        margin_locked=0.0,
        exchange_protected=True,
    )
