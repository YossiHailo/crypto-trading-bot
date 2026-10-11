"""Exponential backoff retries fatal errors immediately."""

from __future__ import annotations

import pytest

from core.retry import call_with_retries


class RetryableError(Exception):
    pass


class FatalError(Exception):
    pass


@pytest.mark.asyncio
async def test_retries_then_succeeds() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    async def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("down")
        return "ok"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    result = await call_with_retries(
        fn,
        attempts=3,
        base_delay=0.5,
        retry_exceptions=(RetryableError,),
        fatal_exceptions=(FatalError,),
        sleep=sleep,
    )
    assert result == "ok"
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_fatal_errors_are_not_retried() -> None:
    calls = {"n": 0}

    async def fn() -> None:
        calls["n"] += 1
        raise FatalError("no funds")

    with pytest.raises(FatalError):
        await call_with_retries(
            fn,
            attempts=4,
            base_delay=0.5,
            retry_exceptions=(RetryableError,),
            fatal_exceptions=(FatalError,),
            sleep=_no_sleep,
        )
    assert calls["n"] == 1


async def _no_sleep(_delay: float) -> None:
    return None
