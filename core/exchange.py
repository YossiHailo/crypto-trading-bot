"""CCXT client factory.

Market data uses a public client. Live execution uses a second, authenticated
client. Both enable CCXT's ``enableRateLimit`` so the process backs off instead
of tripping exchange bans during a 24/7 loop.
"""

from __future__ import annotations

from typing import Any

import ccxt.async_support as ccxt_async

from config.settings import Settings
from core.errors import LiveTradingNotConfirmedError


def build_ccxt_exchange(settings: Settings, *, authenticated: bool) -> Any:
    """Construct a CCXT async exchange. The caller must ``await exchange.close()``."""
    exchange_cls = getattr(ccxt_async, settings.exchange_id, None)
    if exchange_cls is None:
        raise LiveTradingNotConfirmedError(f"CCXT has no exchange named {settings.exchange_id}")
    config: dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": int(settings.request_timeout_seconds * 1000),
        "options": {"defaultType": settings.market_type},
    }
    if authenticated:
        if not settings.exchange_api_key or not settings.exchange_api_secret:
            raise LiveTradingNotConfirmedError(
                "Authenticated exchange client requires API key and secret."
            )
        config["apiKey"] = settings.exchange_api_key
        config["secret"] = settings.exchange_api_secret
        if settings.exchange_password:
            config["password"] = settings.exchange_password
    exchange = exchange_cls(config)
    if settings.exchange_sandbox:
        exchange.set_sandbox_mode(True)
    return exchange


def apply_public_url(exchange: Any, url: str) -> None:
    """Point a CCXT client's public REST API at ``url`` and load spot markets only."""
    api = getattr(exchange, "urls", {}).get("api")
    if isinstance(api, dict) and "public" in api:
        api["public"] = url.rstrip("/")
    # Spot candles only: avoids loading futures markets from hosts that may be blocked.
    options = dict(getattr(exchange, "options", {}) or {})
    current = options.get("fetchMarkets")
    # CCXT changed this option's shape over time: a list, or {"types": [...]}.
    options["fetchMarkets"] = {**current, "types": ["spot"]} if isinstance(current, dict) else ["spot"]
    exchange.options = options


def build_market_data_exchanges(settings: Settings) -> list[Any]:
    """Public clients for candles: the configured exchange, then an optional fallback."""
    primary = build_ccxt_exchange(settings, authenticated=False)
    if settings.market_data_url:
        apply_public_url(primary, settings.market_data_url)
    clients = [primary]
    if settings.market_data_fallback and settings.market_data_fallback != settings.exchange_id:
        cls = getattr(ccxt_async, settings.market_data_fallback, None)
        if cls is not None:
            clients.append(cls({"enableRateLimit": True, "timeout": int(settings.request_timeout_seconds * 1000)}))
    return clients
