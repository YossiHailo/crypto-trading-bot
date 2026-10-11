"""Historical candles (paginated CCXT download with a CSV cache) and Fear & Greed history."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from core.timeframes import timeframe_to_ms

logger = logging.getLogger("trading.backtest.data")


def cache_path(cache_dir: str | Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-").replace(":", "-")
    return Path(cache_dir) / f"{exchange_id}_{safe}_{timeframe}.csv"


def load_csv(path: str | Path) -> list[list[float]]:
    """Read ``timestamp,open,high,low,close,volume`` rows (header optional, ms timestamps)."""
    rows: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.reader(handle):
            if not record or not record[0].strip().lstrip("-").isdigit():
                continue
            rows.append([int(record[0])] + [float(value) for value in record[1:6]])
    rows.sort(key=lambda row: row[0])
    deduped: list[list[float]] = []
    for row in rows:
        if deduped and deduped[-1][0] == row[0]:
            deduped[-1] = row
        else:
            deduped.append(row)
    return deduped


def save_csv(path: str | Path, rows: list[list[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)


async def download_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    *,
    cache_dir: str | Path = "data/history",
    market_type: str = "spot",
) -> list[list[float]]:
    """Fetch closed candles in [since, until), reusing and extending the CSV cache."""
    import ccxt.async_support as ccxt_async

    path = cache_path(cache_dir, exchange_id, symbol, timeframe)
    cached = load_csv(path) if path.exists() else []
    step = timeframe_to_ms(timeframe)
    have = {int(row[0]) for row in cached}
    need_start = since_ms
    if cached and int(cached[0][0]) <= since_ms:
        need_start = int(cached[-1][0]) + step
    exchange = getattr(ccxt_async, exchange_id)({"enableRateLimit": True, "options": {"defaultType": market_type}})
    fetched: list[list[float]] = []
    try:
        cursor = need_start
        while cursor < until_ms:
            batch = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
            if not batch:
                break
            fetched.extend(row for row in batch if int(row[0]) not in have)
            last = int(batch[-1][0])
            if last + step <= cursor:
                break
            cursor = last + step
            logger.info("%s %s: downloaded up to %s", symbol, timeframe, datetime.fromtimestamp(last / 1000, tz=timezone.utc))
            await asyncio.sleep(0)
    finally:
        await exchange.close()
    merged = {int(row[0]): row for row in cached}
    for row in fetched:
        merged[int(row[0])] = [int(row[0])] + [float(v) for v in row[1:6]]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = [merged[key] for key in sorted(merged) if key + step <= now_ms]  # closed candles only
    save_csv(path, rows)
    return [row for row in rows if since_ms <= int(row[0]) < until_ms]


async def fear_greed_history(url: str = "https://api.alternative.me/fng/?limit=0&format=json") -> dict[date, int]:
    """Daily Crypto Fear & Greed values (alternative.me keeps the full history)."""
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            payload = json.loads(await response.text())
    out: dict[date, int] = {}
    for row in payload.get("data", []):
        try:
            day = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc).date()
            out[day] = int(row["value"])
        except (KeyError, ValueError, TypeError):
            continue
    return out
