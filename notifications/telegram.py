"""Telegram Bot API notifications via aiohttp.

``send`` returns immediately when ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID``
is empty, so the trading loop never depends on Telegram being configured.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import aiohttp

from core.models import ClosedTrade, OrderResult, RiskDecision
from core.redact import redact

logger = logging.getLogger("trading.notify")

_MAX_LEN = 4000


class TelegramNotifier:
    """Posts trade, error, and daily-summary messages. Disabled without credentials."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        timeout_seconds: float = 10,
        transport: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._session: aiohttp.ClientSession | None = None
        self.enabled = bool(self.token and self.chat_id)

    async def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        text = redact(text)
        payload = text if len(text) <= _MAX_LEN else text[: _MAX_LEN - 3] + "..."
        try:
            if self._transport is not None:
                await self._transport(payload)
                return True
            session = await self._session_get()
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            async with session.post(
                url,
                json={"chat_id": self.chat_id, "text": payload, "disable_web_page_preview": True},
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning("Telegram returned %s: %s", response.status, redact(body[:300]))
                    return False
            return True
        except Exception as exc:
            # The request URL contains the bot token; never log the raw exception.
            logger.warning("Telegram send failed: %s", redact(exc))
            return False

    async def info(self, text: str) -> bool:
        return await self.send(text)

    async def error(self, message: str) -> bool:
        return await self.send(f"Error\n{message}")

    async def trade_opened(self, decision: RiskDecision, result: OrderResult) -> bool:
        text = (
            f"Trade opened {decision.action.value} {decision.symbol}\n"
            f"Size: {result.amount:.8g}\n"
            f"Entry: {result.price:.8g}\n"
            f"Stop: {decision.stop_loss:.8g}\n"
            f"Target: {decision.take_profit:.8g}\n"
            f"Confidence: {decision.confidence:.2f}\n"
            f"{decision.reasoning}"
        )
        return await self.send(text)

    async def trade_closed(self, trade: ClosedTrade) -> bool:
        text = (
            f"Trade closed {trade.symbol} {trade.side}\n"
            f"Size: {trade.size:.8g}\n"
            f"Entry: {trade.entry:.8g}\n"
            f"Exit: {trade.exit_price:.8g}\n"
            f"PnL: {trade.pnl:.4f}\n"
            f"Reason: {trade.reason.value}"
        )
        return await self.send(text)

    async def daily_summary(self, text: str) -> bool:
        return await self.send(text)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
