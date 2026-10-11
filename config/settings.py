"""Environment-driven settings.

Live trading is refused unless ``PAPER_TRADING`` is false and
``LIVE_TRADING_CONFIRM`` is true. Missing variables keep the paper defaults.

Every risk knob has a hard ceiling here so a typo in ``.env`` (for example
``RISK_PER_TRADE=1`` instead of ``0.01``) fails at startup instead of trading.
"""

from __future__ import annotations

import ipaddress
import json
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.errors import LiveTradingNotConfirmedError

STRATEGIES = {"trend", "legacy"}
# Bars each strategy needs before it can score (kept here so settings can
# reject an OHLCV_LIMIT that would silently produce HOLD forever).
STRATEGY_MIN_BARS = {"trend": 210, "legacy": 59}


def _split_csv(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("expected a list")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def _is_loopback(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    """All knobs the agents read. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    paper_trading: bool = True
    live_trading_confirm: bool = False

    exchange_id: str = "binance"
    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    exchange_password: str = ""
    exchange_sandbox: bool = False
    market_type: str = "spot"
    leverage: float = 1.0

    strategy: str = "trend"
    symbols: str = "BTC/USDT,ETH/USDT"
    timeframe: str = "4h"
    loop_interval_minutes: float = 15
    ohlcv_limit: int = 300

    risk_per_trade: float = 0.01
    reward_risk_ratio: float = 3.0
    min_reward_risk: float = 1.5
    min_confidence: float = 0.55
    max_open_positions: int = 3
    daily_max_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    cooldown_minutes: float = 60
    atr_multiplier: float = 2.5
    use_atr_stops: bool = True
    fixed_stop_pct: float = 0.03
    # Per-position notional cap vs equity, and the cap on all open notional.
    max_notional_pct: float = 0.5
    max_total_exposure_pct: float = 1.0
    allow_short: bool = False
    order_type: str = "market"
    fee_rate: float = 0.001
    slippage_pct: float = 0.0005
    cash_utilization: float = 0.98
    min_notional: float = 10.0
    starting_balance: float = 10_000.0
    # Paper only: a simulated bank transfer added every month on DEPOSIT_DAY (00:00 UTC).
    monthly_deposit: float = 0.0
    deposit_day: int = 1
    quote_currency: str = "USDT"
    # Market data source overrides. Binance blocks some regions (HTTP 451), for
    # example US cloud runners; its public mirror https://data-api.binance.vision/api/v3
    # serves the same candles. MARKET_DATA_FALLBACK (e.g. kraken) is tried when the
    # primary source fails.
    market_data_url: str = ""
    market_data_fallback: str = ""
    pending_order_ttl_minutes: float = 60

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    cryptopanic_token: str = ""
    fear_greed_url: str = "https://api.alternative.me/fng/?limit=1"
    rss_feeds: str = (
        "https://www.coindesk.com/arc/outboundfeeds/rss/,"
        "https://cointelegraph.com/rss"
    )

    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    dashboard_token: str = ""

    log_level: str = "INFO"
    data_dir: str = "data"
    log_dir: str = "logs"
    request_timeout_seconds: float = 15
    retry_attempts: int = 3
    retry_base_delay: float = 0.5
    order_poll_attempts: int = 5
    order_poll_delay: float = 0.4
    client_order_prefix: str = "ctb"
    decisions_retention_days: int = 90

    @field_validator("exchange_id")
    @classmethod
    def _exchange_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"binance", "bybit"}:
            raise ValueError("EXCHANGE_ID must be binance or bybit")
        return normalized

    @field_validator("market_type")
    @classmethod
    def _market_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"spot", "swap"}:
            raise ValueError("MARKET_TYPE must be spot or swap")
        return normalized

    @field_validator("strategy")
    @classmethod
    def _strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in STRATEGIES:
            raise ValueError(f"STRATEGY must be one of {sorted(STRATEGIES)}")
        return normalized

    @field_validator("order_type")
    @classmethod
    def _order_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"market", "limit"}:
            raise ValueError("ORDER_TYPE must be market or limit")
        return normalized

    @field_validator("log_level")
    @classmethod
    def _log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL is invalid")
        return normalized

    @field_validator("timeframe", "quote_currency")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @field_validator("timeframe")
    @classmethod
    def _timeframe(cls, value: str) -> str:
        from core.timeframes import timeframe_to_ms

        timeframe_to_ms(value)  # raises ValueError on an unknown unit
        return value

    @field_validator("client_order_prefix")
    @classmethod
    def _prefix(cls, value: str) -> str:
        text = value.strip()
        if not text.isalnum() or not 2 <= len(text) <= 8:
            raise ValueError("CLIENT_ORDER_PREFIX must be 2-8 letters or digits")
        return text

    @field_validator("risk_per_trade")
    @classmethod
    def _risk_per_trade(cls, value: float) -> float:
        # 5% per trade is already aggressive. Anything above is almost always a typo.
        if not 0 < value <= 0.05:
            raise ValueError("RISK_PER_TRADE must be in (0, 0.05]. 0.01 means 1% of equity.")
        return value

    @field_validator("daily_max_loss_pct", "fixed_stop_pct", "cash_utilization", "max_drawdown_pct")
    @classmethod
    def _open_unit_interval(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("value must be in (0, 1]")
        return value

    @field_validator("max_notional_pct", "max_total_exposure_pct")
    @classmethod
    def _notional_cap(cls, value: float) -> float:
        # Values above 1 only make sense on swap with leverage. Spot is cash-capped anyway.
        if not 0 < value <= 10:
            raise ValueError("notional caps must be in (0, 10]")
        return value

    @field_validator("leverage")
    @classmethod
    def _leverage(cls, value: float) -> float:
        if not 1 <= value <= 10:
            raise ValueError("LEVERAGE must be in [1, 10]")
        return value

    @field_validator("min_confidence")
    @classmethod
    def _confidence(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("MIN_CONFIDENCE must be in [0, 1]")
        return value

    @field_validator("fee_rate")
    @classmethod
    def _fee_rate(cls, value: float) -> float:
        if not 0 <= value < 0.05:
            raise ValueError("FEE_RATE must be in [0, 0.05)")
        return value

    @field_validator("slippage_pct")
    @classmethod
    def _slippage(cls, value: float) -> float:
        if not 0 <= value <= 0.02:
            raise ValueError("SLIPPAGE_PCT must be in [0, 0.02]")
        return value

    @field_validator("reward_risk_ratio", "min_reward_risk", "atr_multiplier")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator(
        "loop_interval_minutes",
        "request_timeout_seconds",
        "retry_base_delay",
        "order_poll_delay",
        "pending_order_ttl_minutes",
    )
    @classmethod
    def _positive_time(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("cooldown_minutes", "min_notional")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value must be >= 0")
        return value

    @field_validator("starting_balance")
    @classmethod
    def _balance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("STARTING_BALANCE must be positive")
        return value

    @field_validator("ohlcv_limit")
    @classmethod
    def _ohlcv_limit(cls, value: int) -> int:
        if value < 60:
            raise ValueError("OHLCV_LIMIT must be >= 60 so indicators can warm up")
        if value > 1000:
            raise ValueError("OHLCV_LIMIT must be <= 1000 (one exchange request)")
        return value

    @field_validator("monthly_deposit")
    @classmethod
    def _deposit(cls, value: float) -> float:
        if value < 0:
            raise ValueError("MONTHLY_DEPOSIT must be >= 0")
        return value

    @field_validator("deposit_day")
    @classmethod
    def _deposit_day(cls, value: int) -> int:
        if not 1 <= value <= 31:
            raise ValueError("DEPOSIT_DAY must be 1-31 (short months use their last day)")
        return value

    @field_validator("market_data_fallback")
    @classmethod
    def _fallback(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized and normalized not in {"kraken", "bybit", "binance", "okx", "coinbase"}:
            raise ValueError("MARKET_DATA_FALLBACK must be empty or one of kraken, bybit, binance, okx, coinbase")
        return normalized

    @field_validator("dashboard_port")
    @classmethod
    def _port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("DASHBOARD_PORT must be a valid TCP port")
        return value

    @field_validator("max_open_positions", "retry_attempts", "order_poll_attempts", "decisions_retention_days")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be >= 1")
        return value

    @field_validator(
        "exchange_api_key",
        "exchange_api_secret",
        "exchange_password",
        "telegram_bot_token",
        "telegram_chat_id",
        "cryptopanic_token",
        "dashboard_token",
        "dashboard_host",
    )
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        if not self.symbol_list():
            raise ValueError("SYMBOLS must contain at least one pair")
        need = STRATEGY_MIN_BARS[self.strategy] + 2
        if self.ohlcv_limit < need:
            raise ValueError(
                f"STRATEGY={self.strategy} needs OHLCV_LIMIT >= {need} "
                "(indicator warm-up plus the dropped unclosed candle)"
            )
        if self.monthly_deposit and not self.paper_trading:
            raise ValueError("MONTHLY_DEPOSIT is a paper-trading simulation. In live mode deposit real money on the exchange.")
        if self.market_type == "spot" and self.leverage != 1:
            raise ValueError("LEVERAGE only applies to MARKET_TYPE=swap. Keep it at 1 on spot.")
        if self.dashboard_enabled and not _is_loopback(self.dashboard_host) and not self.dashboard_token:
            raise ValueError(
                "DASHBOARD_TOKEN is required when DASHBOARD_HOST is not a loopback address. "
                "The dashboard can pause the bot and close positions."
            )
        return self

    def symbol_list(self) -> list[str]:
        return _split_csv(self.symbols)

    def rss_feed_list(self) -> list[str]:
        return _split_csv(self.rss_feeds)

    def secret_values(self) -> list[str]:
        """Values the log redactor must never print."""
        return [
            value
            for value in (
                self.exchange_api_key,
                self.exchange_api_secret,
                self.exchange_password,
                self.telegram_bot_token,
                self.cryptopanic_token,
                self.dashboard_token,
            )
            if value
        ]

    def public_summary(self) -> dict[str, Any]:
        """Non-secret configuration shown on the dashboard."""
        return {
            "mode": "paper" if self.paper_trading else "live",
            "exchange": self.exchange_id,
            "market_type": self.market_type,
            "sandbox": self.exchange_sandbox,
            "strategy": self.strategy,
            "symbols": self.symbol_list(),
            "timeframe": self.timeframe,
            "loop_interval_minutes": self.loop_interval_minutes,
            "risk_per_trade": self.risk_per_trade,
            "reward_risk_ratio": self.reward_risk_ratio,
            "max_open_positions": self.max_open_positions,
            "max_notional_pct": self.max_notional_pct,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "daily_max_loss_pct": self.daily_max_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "allow_short": self.allow_short,
            "leverage": self.leverage,
            "fee_rate": self.fee_rate,
            "slippage_pct": self.slippage_pct,
            "quote_currency": self.quote_currency,
            "starting_balance": self.starting_balance,
            "monthly_deposit": self.monthly_deposit,
            "deposit_day": self.deposit_day,
            "cooldown_minutes": self.cooldown_minutes,
        }


def assert_trading_mode(settings: Settings) -> str:
    """Return ``paper`` or ``live``.

    Going live requires an explicit opt-out of paper mode and a second flag
    so a single mistaken environment variable cannot send real orders.
    """
    if settings.paper_trading:
        return "paper"
    if not settings.live_trading_confirm:
        raise LiveTradingNotConfirmedError(
            "Refusing to start live trading. PAPER_TRADING=false also requires "
            "LIVE_TRADING_CONFIRM=true. Leaving the confirm flag unset keeps the "
            "process from sending real orders."
        )
    if not settings.exchange_api_key or not settings.exchange_api_secret:
        raise LiveTradingNotConfirmedError(
            "Live trading requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET. "
            "Create trade-only keys with withdrawals disabled."
        )
    return "live"
