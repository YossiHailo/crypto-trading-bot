"""Timeframe arithmetic and the closed-candle filter.

Exchanges return the candle that is still forming as the last OHLCV row.
Scoring that row means the signal can flip several times inside one bar
(the loop runs every 15 minutes on hourly or 4h candles). Every decision in
this bot is taken on closed bars only. The forming bar is still useful for
the current price and for the intrabar high/low used by the paper stops.
"""

from __future__ import annotations

import re

_UNIT_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}
_PATTERN = re.compile(r"^(\d+)([mhdw])$")


def timeframe_to_ms(timeframe: str) -> int:
    """``"15m"`` -> 900000. Months are rejected because their length varies."""
    match = _PATTERN.match(timeframe.strip())
    if not match:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Use e.g. 15m, 1h, 4h, 1d.")
    count = int(match.group(1))
    if count <= 0:
        raise ValueError("timeframe count must be positive")
    return count * _UNIT_MS[match.group(2)]


def split_closed(rows: list[list[float]], timeframe: str, now_ms: int) -> tuple[list[list[float]], list[float] | None]:
    """Return ``(closed_rows, forming_row)``.

    A row is closed when ``open_time + timeframe <= now``. Only the last row
    can be open, so this is O(1).
    """
    if not rows:
        return [], None
    step = timeframe_to_ms(timeframe)
    last = rows[-1]
    if int(last[0]) + step > now_ms:
        return rows[:-1], last
    return rows, None


def bars_per_year(timeframe: str) -> float:
    """Annualization factor for Sharpe/Sortino. Crypto trades 365 days a year."""
    return 365 * 86_400_000 / timeframe_to_ms(timeframe)
