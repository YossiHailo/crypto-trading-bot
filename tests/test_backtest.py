"""The offline replay finishes and keeps a positive equity figure."""

from __future__ import annotations

import pytest

from scripts.backtest_lite import run_backtest


@pytest.mark.asyncio
async def test_backtest_lite_completes(tmp_path) -> None:
    result = await run_backtest(bars=90, min_history=70, data_dir=tmp_path, seed=7)
    assert result["cycles"] == 21
    assert result["equity"] > 0
    assert result["closed_trades"] >= 0
