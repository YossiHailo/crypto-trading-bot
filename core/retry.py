"""Small async retry helper.

Network blips are retried with exponential backoff. Fatal errors (insufficient
funds, invalid orders, authentication) propagate on the first attempt so we
do not hammer the exchange with a request it has already rejected.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def call_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
    retry_exceptions: tuple[type[BaseException], ...],
    fatal_exceptions: tuple[type[BaseException], ...] = (),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``fn`` until it succeeds, a fatal error is raised, or attempts run out.

    ``fn`` must be a zero-argument callable that returns a new awaitable on
    every attempt. Do not pass an already-created coroutine.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    delay = base_delay
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except fatal_exceptions:
            raise
        except retry_exceptions as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            await sleep(delay)
            delay *= 2
    # Unreachable, but keeps the type checker satisfied if attempts is 0-checked.
    assert last_error is not None
    raise last_error
