"""Structural interfaces for the four agents.

Concrete classes live beside this module. The orchestrator depends on these
methods, not on a framework, so tests can pass plain stubs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.models import (
    Bar,
    MarkResult,
    OrderResult,
    PortfolioView,
    RiskDecision,
    SentimentSignal,
    TechnicalSignal,
)


class MarketAnalyzer(Protocol):
    async def analyze(self, symbol: str) -> TechnicalSignal:
        """Return a technical signal for one symbol."""

    async def close(self) -> None:
        """Release network clients."""


class SentimentAnalyzer(Protocol):
    async def evaluate(self) -> SentimentSignal:
        """Return a market-wide sentiment score. Must not raise on source outages."""

    async def close(self) -> None:
        """Release network clients."""


class DecisionMaker(Protocol):
    def decide(
        self,
        technical: TechnicalSignal,
        sentiment: SentimentSignal,
        portfolio: PortfolioView,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Combine signals and account state into one risk decision."""


class OrderExecutor(Protocol):
    async def execute(
        self,
        decision: RiskDecision,
        market_price: float | None = None,
    ) -> OrderResult:
        """Place, simulate, or skip an order."""

    async def mark_prices(
        self,
        prices: dict[str, float],
        bars: dict[str, tuple[Bar, ...]] | None = None,
    ) -> MarkResult:
        """Update resting orders and trigger stop-loss / take-profit."""

    async def close_all(self, prices: dict[str, float]) -> list[OrderResult]:
        """Operator emergency exit: cancel resting entries and close every position."""

    async def reconcile(self, symbols: list[str]) -> list[str]:
        """Compare local state with the exchange at startup. Returns findings."""

    def portfolio_view(self, marks: dict[str, float] | None = None) -> PortfolioView:
        """Snapshot used by the risk agent."""

    async def close(self) -> None:
        """Release clients and the state database."""
