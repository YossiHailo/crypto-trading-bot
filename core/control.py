"""Runtime status and operator controls shared by the main loop and the dashboard.

Everything runs on one asyncio event loop, so plain attributes are safe. The
``cycle_lock`` makes sure an operator action such as "close all positions"
never interleaves with a decision cycle that is halfway through.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.models import CycleReport, utcnow


@dataclass
class BotStatus:
    mode: str = "paper"
    started_at: datetime = field(default_factory=utcnow)
    cycle_count: int = 0
    cycle_running: bool = False
    last_cycle_started: datetime | None = None
    last_cycle_finished: datetime | None = None
    next_cycle_at: datetime | None = None
    last_prices: dict[str, float] = field(default_factory=dict)
    last_signals: list[dict[str, Any]] = field(default_factory=list)
    last_sentiment: dict[str, Any] | None = None
    last_errors: list[str] = field(default_factory=list)
    stopping: bool = False


class Controller:
    def __init__(self, mode: str = "paper") -> None:
        self.status = BotStatus(mode=mode)
        self.stop_event = asyncio.Event()
        self.wake_event = asyncio.Event()
        self.cycle_lock = asyncio.Lock()

    def request_stop(self) -> None:
        self.status.stopping = True
        self.stop_event.set()
        self.wake_event.set()

    def request_cycle(self) -> None:
        self.wake_event.set()

    def record_report(self, report: CycleReport, prices: dict[str, float]) -> None:
        status = self.status
        status.cycle_count += 1
        status.last_cycle_started = report.started_at
        status.last_cycle_finished = report.finished_at
        status.last_prices = dict(prices)
        status.last_errors = list(report.errors)[-30:]
        status.last_signals = [
            {
                "symbol": signal.symbol,
                "action": signal.action.value,
                "confidence": round(signal.confidence, 4),
                "price": signal.price,
                "atr": signal.atr,
                "strategy": signal.strategy,
                "exit_long": signal.exit_long,
                "exit_short": signal.exit_short,
                "reasoning": signal.reasoning,
                "indicators": {k: round(float(v), 8) for k, v in signal.indicators.items()},
            }
            for signal in report.signals
        ]
        if report.sentiment is not None:
            status.last_sentiment = {
                "bias": report.sentiment.bias.value,
                "score": round(report.sentiment.score, 4),
                "fear_greed": report.sentiment.fear_greed,
                "sources": list(report.sentiment.sources_used),
                "reasoning": report.sentiment.reasoning,
            }
