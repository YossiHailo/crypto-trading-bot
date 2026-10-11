"""Fundamental and sentiment agent.

Two public sources are combined into one score from -1 (bearish) to +1 (bullish):

* Crypto Fear & Greed (alternative.me). 0 maps to -1, 50 maps to 0, 100 maps to +1.
* Headlines from CryptoPanic when ``CRYPTOPANIC_TOKEN`` is set, otherwise the
  CoinDesk and Cointelegraph RSS feeds. A small lexicon counts bullish and
  bearish phrases per headline. A negating verb ("delays", "rejects", ...)
  turns a headline's bullish hits bearish, so "SEC delays spot ETF approval"
  is no longer scored as good news. Duplicate titles are counted once. Thin
  evidence is shrunk toward zero so one headline cannot veto the whole book.

Limits worth knowing: Fear & Greed updates once a day, so it is a slow regime
gauge, not an intraday signal. Headlines are market-wide (not per symbol) and
are usually priced in long before a 15-minute loop reads them. Treat this
agent as a veto/brake, not as a source of edge.

If every source fails, the result is NEUTRAL with score 0 and the bot continues.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from config.settings import Settings
from core.models import SentimentBias, SentimentSignal, clamp_score, neutral_sentiment
from core.redact import redact

logger = logging.getLogger("trading.sentiment")

# Multi-word phrases are matched as substrings. Single words use boundaries
# so "ban" does not fire inside "bank" and "hack" does not require "hacker".
BULLISH_PHRASES: tuple[str, ...] = (
    "record high",
    "all-time high",
    "all time high",
    "breakout",
    "breaks out",
    "etf approval",
    "spot etf",
    "partnership",
    "institutional",
    "rate cut",
    "bullish",
    "surge",
    "surged",
    "surges",
    "rally",
    "rallies",
    "soars",
    "soared",
    "soar",
    "adoption",
    "approval",
    "inflows",
    "upgrade",
    "outperform",
    "accumulation",
)
BEARISH_PHRASES: tuple[str, ...] = (
    "exploit",
    "hacked",
    "hack",
    "hacker",
    "lawsuit",
    "sec charges",
    "ban",
    "banned",
    "crash",
    "crashes",
    "crashed",
    "plunge",
    "plunges",
    "plunged",
    "selloff",
    "sell-off",
    "bearish",
    "fraud",
    "bankrupt",
    "bankruptcy",
    "liquidation",
    "liquidations",
    "outflows",
    "downgrade",
    "investigation",
    "collapsed",
    "collapse",
    "dump",
    "dumped",
    "delisting",
)

NEGATORS: tuple[str, ...] = (
    "delay",
    "delays",
    "delayed",
    "postpones",
    "postponed",
    "reject",
    "rejects",
    "rejected",
    "denies",
    "denied",
    "blocks",
    "blocked",
    "halts",
    "halted",
    "fails",
    "failed",
    "not",
    "no",
)

_USER_AGENT = "crypto-trading-bot/1.1 (paper-default; local research)"
_HEADLINE_FULL_EVIDENCE = 4.0


def _contains(text: str, phrase: str) -> bool:
    if " " in phrase or "-" in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def score_headlines(headlines: list[str]) -> tuple[float, int, int]:
    """Return ``(score, bullish_hits, bearish_hits)``.

    Hits are counted per distinct headline. The raw hit ratio is multiplied
    by ``min(1, hits / 4)`` so a single keyword moves the score by 0.25 and
    four agreeing hits are required for a full -1 or +1.
    """
    bullish = 0
    bearish = 0
    seen: set[str] = set()
    for headline in headlines:
        text = " ".join(headline.lower().split())
        if not text or text in seen:
            continue
        seen.add(text)
        bulls = sum(1 for phrase in BULLISH_PHRASES if _contains(text, phrase))
        bears = sum(1 for phrase in BEARISH_PHRASES if _contains(text, phrase))
        if bulls and any(_contains(text, word) for word in NEGATORS):
            bears += bulls
            bulls = 0
        bullish += bulls
        bearish += bears
    hits = bullish + bearish
    if hits == 0:
        return 0.0, 0, 0
    raw = (bullish - bearish) / hits
    score = raw * min(1.0, hits / _HEADLINE_FULL_EVIDENCE)
    return clamp_score(score), bullish, bearish


def score_fear_greed(value: int) -> float:
    clamped = max(0, min(100, int(value)))
    return (clamped - 50) / 50.0


def fear_greed_label(value: int) -> str:
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


def parse_fear_greed(payload: dict[str, Any]) -> tuple[int, str]:
    row = payload["data"][0]
    value = int(row["value"])
    label = str(row.get("value_classification") or fear_greed_label(value))
    return value, label


def parse_cryptopanic(payload: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for row in payload.get("results", []):
        title = row.get("title")
        if title:
            titles.append(str(title).strip())
    return titles


def parse_rss(xml_text: str) -> list[str]:
    """Pull ``<title>`` text out of each RSS item, ignoring XML namespaces."""
    root = ET.fromstring(xml_text)
    titles: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) != "item":
            continue
        for child in list(element):
            if _local_name(child.tag) == "title" and child.text:
                titles.append(child.text.strip())
    return titles


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def build_sentiment(
    *,
    fear_greed: int | None,
    fear_greed_label_text: str | None,
    headlines: list[str] | None,
    errors: list[str],
) -> SentimentSignal:
    """Average every source that actually returned data."""
    parts: list[float] = []
    used: list[str] = []
    headline_score: float | None = None
    notes: list[str] = []

    if fear_greed is not None:
        fg_score = score_fear_greed(fear_greed)
        parts.append(fg_score)
        used.append("fear_greed")
        label = fear_greed_label_text or fear_greed_label(fear_greed)
        notes.append(f"Fear & Greed {fear_greed} ({label}) -> {fg_score:+.2f}")

    if headlines is not None:
        headline_score, bullish, bearish = score_headlines(headlines)
        parts.append(headline_score)
        used.append("headlines")
        notes.append(
            f"Headlines {len(headlines)} items, bullish hits {bullish}, "
            f"bearish hits {bearish} -> {headline_score:+.2f}"
        )

    if not parts:
        detail = "; ".join(errors) if errors else "no sentiment data"
        return neutral_sentiment(detail)

    score = clamp_score(sum(parts) / len(parts))
    if score > 0.15:
        bias = SentimentBias.BULLISH
    elif score < -0.15:
        bias = SentimentBias.BEARISH
    else:
        bias = SentimentBias.NEUTRAL
    if errors:
        notes.append("Partial outage: " + "; ".join(errors))
    reasoning = f"{bias.value} score {score:+.2f}. " + ". ".join(notes)
    return SentimentSignal(
        bias=bias,
        score=score,
        reasoning=reasoning,
        fear_greed=fear_greed,
        headline_score=headline_score,
        sources_used=tuple(used),
    )


class SentimentAgent:
    """Fetches Fear & Greed and headlines. Network failures become NEUTRAL."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_get: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self.settings = settings
        self._http_get = http_get
        self._session: aiohttp.ClientSession | None = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _request_text(self, url: str) -> str:
        if self._http_get is not None:
            return await self._http_get(url)
        session = await self._session_get()
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json, application/xml, text/xml, */*"}
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.text()

    async def fetch_fear_greed(self) -> tuple[int, str]:
        import json

        body = await self._request_text(self.settings.fear_greed_url)
        return parse_fear_greed(json.loads(body))

    async def fetch_headlines(self) -> list[str]:
        import json

        errors: list[str] = []
        if self.settings.cryptopanic_token:
            # The token has to travel in the query string. ``redact`` masks it in errors.
            url = (
                "https://cryptopanic.com/api/v1/posts/"
                f"?auth_token={self.settings.cryptopanic_token}&public=true&kind=news"
            )
            try:
                payload = json.loads(await self._request_text(url))
                titles = parse_cryptopanic(payload)
                if titles:
                    return titles[:30]
                errors.append("cryptopanic returned no titles")
            except Exception as exc:
                errors.append(f"cryptopanic: {redact(exc)}")
                logger.warning("CryptoPanic failed: %s", redact(exc))

        titles: list[str] = []
        for feed in self.settings.rss_feed_list():
            try:
                titles.extend(parse_rss(await self._request_text(feed)))
            except Exception as exc:
                errors.append(f"rss {feed}: {redact(exc)}")
                logger.warning("RSS feed failed %s: %s", feed, redact(exc))
        if titles:
            return titles[:30]
        if errors:
            raise RuntimeError("; ".join(errors))
        return []

    async def evaluate(self) -> SentimentSignal:
        import asyncio

        fear_task = asyncio.create_task(self._safe_fear_greed())
        headline_task = asyncio.create_task(self._safe_headlines())
        fear, headlines = await asyncio.gather(fear_task, headline_task)
        fear_value, fear_label, fear_error = fear
        headline_values, headline_error = headlines
        errors = [item for item in (fear_error, headline_error) if item]
        signal = build_sentiment(
            fear_greed=fear_value,
            fear_greed_label_text=fear_label,
            headlines=headline_values,
            errors=errors,
        )
        logger.info(
            "sentiment %s score=%+.2f sources=%s",
            signal.bias.value,
            signal.score,
            ",".join(signal.sources_used) or "none",
        )
        return signal

    async def _safe_fear_greed(self) -> tuple[int | None, str | None, str | None]:
        try:
            value, label = await self.fetch_fear_greed()
            return value, label, None
        except Exception as exc:
            logger.warning("Fear & Greed failed: %s", redact(exc))
            return None, None, f"fear_greed: {redact(exc)}"

    async def _safe_headlines(self) -> tuple[list[str] | None, str | None]:
        try:
            return await self.fetch_headlines(), None
        except Exception as exc:
            logger.warning("Headlines failed: %s", redact(exc))
            return None, f"headlines: {redact(exc)}"

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
