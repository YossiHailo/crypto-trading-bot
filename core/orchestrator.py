"""One decision cycle.

Order of work:

1. Fetch candles for every symbol concurrently and score the closed bars.
2. Mark open positions (stops, targets, resting entries, exchange fills).
3. Roll the UTC day and update the equity high-water mark (kill switch).
4. Score sentiment once for the whole book.
5. Ask the risk agent about each symbol, record the decision, execute it.
6. Save an equity snapshot and write the heartbeat.

Symbols are isolated: a raised exception, a timeout, or a rejected order is
recorded and the loop moves on. If sentiment throws, the cycle uses NEUTRAL.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import DecisionMaker, MarketAnalyzer, OrderExecutor, SentimentAnalyzer
from config.settings import Settings
from core.models import Bar, CycleReport, RiskAction, TechnicalSignal, neutral_sentiment, utcnow
from core.redact import redact

logger = logging.getLogger("trading.orchestrator")


def _one_line(exc: BaseException) -> str:
    """Exchange errors often include a pretty-printed body. Keep the log line single."""
    return redact(" ".join(str(exc).split()))


async def _notify(notifier: Any, method: str, *args: object) -> None:
    try:
        await getattr(notifier, method)(*args)
    except Exception:
        logger.exception("Notifier %s failed", method)


def _write_heartbeat(data_dir: str) -> None:
    try:
        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "heartbeat").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    except OSError:
        logger.exception("Could not write heartbeat")


def _event(store: Any, level: str, message: str) -> None:
    if store is None or not hasattr(store, "add_event"):
        return
    try:
        store.add_event(level, message)
    except Exception:
        logger.debug("Could not store event", exc_info=True)


async def run_cycle(
    *,
    symbols: list[str],
    market: MarketAnalyzer,
    sentiment_agent: SentimentAnalyzer,
    risk: DecisionMaker,
    execution: OrderExecutor,
    notifier: Any,
    settings: Settings,
    stop: asyncio.Event | None = None,
) -> CycleReport:
    """Analyze, protect open positions, then ask risk before any new order."""
    report = CycleReport(started_at=utcnow())
    prices: dict[str, float] = {}
    bars: dict[str, tuple[Bar, ...]] = {}
    timeout = settings.request_timeout_seconds + 5
    store = getattr(execution, "store", None)

    async def analyze(symbol: str) -> TechnicalSignal:
        return await asyncio.wait_for(market.analyze(symbol), timeout=timeout)

    active = [symbol for symbol in symbols if not (stop is not None and stop.is_set())]
    results = await asyncio.gather(*(analyze(symbol) for symbol in active), return_exceptions=True)
    for symbol, outcome in zip(active, results, strict=True):
        if isinstance(outcome, BaseException):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            message = f"{symbol}: market data failed: {_one_line(outcome)}"
            logger.error(message)
            logger.debug("Market data traceback for %s", symbol, exc_info=outcome)
            report.errors.append(message)
            await _notify(notifier, "error", message)
            continue
        report.signals.append(outcome)
        if outcome.price > 0:
            prices[symbol] = outcome.price
        if outcome.recent_bars:
            bars[symbol] = outcome.recent_bars

    try:
        marked = await execution.mark_prices(prices, bars)
    except Exception as exc:
        message = f"Price update failed: {_one_line(exc)}"
        logger.error(message)
        logger.debug("Price update traceback", exc_info=True)
        report.errors.append(message)
        await _notify(notifier, "error", message)
        marked = None
    if marked is not None:
        report.closed_trades.extend(marked.closed)
        report.orders.extend(marked.opened)
        report.errors.extend(redact(error) for error in marked.errors)
        for trade in marked.closed:
            await _notify(notifier, "trade_closed", trade)
        for opened in marked.opened:
            await _notify(notifier, "info", f"Resting order filled {opened.symbol} @ {opened.price:.8g}. {opened.message}")
        for error in marked.errors:
            _event(store, "warning", redact(error))
            await _notify(notifier, "error", redact(error))

    if store is not None and settings.paper_trading and settings.monthly_deposit > 0 and hasattr(store, "apply_monthly_deposits"):
        try:
            for when, amount in store.apply_monthly_deposits(settings.monthly_deposit, settings.deposit_day):
                message = (
                    f"Monthly deposit +{amount:.2f} {settings.quote_currency} (simulated bank transfer due "
                    f"{when:%Y-%m-%d}). Total deposited: {store.total_deposits():.2f}."
                )
                logger.info(message)
                _event(store, "info", message)
                await _notify(notifier, "info", message)
        except Exception as exc:
            report.errors.append(f"Deposit failed: {_one_line(exc)}")

    try:
        if store is not None:
            report.daily_summary = store.roll_day_if_needed(prices)
            if report.daily_summary:
                store.prune(settings.decisions_retention_days)
            halt = store.update_peak_and_check_drawdown(prices)
            if halt:
                message = f"KILL SWITCH: {halt} New entries are blocked until you reset it in the dashboard."
                logger.error(message)
                report.errors.append(message)
                _event(store, "error", message)
                await _notify(notifier, "error", message)
    except Exception as exc:
        message = f"Account roll failed: {_one_line(exc)}"
        logger.error(message)
        logger.debug("Account roll traceback", exc_info=True)
        report.errors.append(message)

    if report.daily_summary:
        logger.info("Daily summary:\n%s", report.daily_summary)
        await _notify(notifier, "daily_summary", report.daily_summary)

    try:
        sentiment = await asyncio.wait_for(sentiment_agent.evaluate(), timeout=timeout * 2)
    except Exception as exc:
        message = f"Sentiment agent failed: {_one_line(exc)}"
        logger.error(message)
        logger.debug("Sentiment traceback", exc_info=True)
        report.errors.append(message)
        sentiment = neutral_sentiment(message)
    report.sentiment = sentiment

    for signal in report.signals:
        if stop is not None and stop.is_set():
            break
        try:
            portfolio = execution.portfolio_view(prices)
            decision = risk.decide(signal, sentiment, portfolio)
        except Exception as exc:
            message = f"{signal.symbol}: risk decision failed: {_one_line(exc)}"
            logger.error(message)
            logger.debug("Risk traceback for %s", signal.symbol, exc_info=True)
            report.errors.append(message)
            await _notify(notifier, "error", message)
            continue
        report.decisions.append(decision)
        if store is not None and hasattr(store, "record_decision"):
            try:
                store.record_decision(signal, decision, sentiment.score)
            except Exception:
                logger.debug("Could not record decision", exc_info=True)
        logger.info("%s %s confidence=%.2f", decision.symbol, decision.action.value, decision.confidence)
        logger.debug("%s decision: %s", decision.symbol, decision.reasoning)
        if decision.action == RiskAction.NO_ACTION:
            continue
        try:
            result = await execution.execute(decision, market_price=signal.price)
        except Exception as exc:
            message = f"{signal.symbol}: execution failed: {_one_line(exc)}"
            logger.error(message)
            logger.debug("Execution traceback for %s", signal.symbol, exc_info=True)
            report.errors.append(message)
            await _notify(notifier, "error", message)
            continue
        report.orders.append(result)
        if not result.success:
            report.errors.append(f"{signal.symbol}: {result.message}")
            _event(store, "warning", f"{signal.symbol}: {result.message}")
            await _notify(notifier, "error", f"{signal.symbol}: {result.message}")
            continue
        if result.opened:
            _event(store, "info", f"Opened {decision.action.value} {signal.symbol} @ {result.price:.8g}")
            await _notify(notifier, "trade_opened", decision, result)
        if result.closed_trade is not None:
            report.closed_trades.append(result.closed_trade)
            await _notify(notifier, "trade_closed", result.closed_trade)

    for trade in report.closed_trades:
        _event(store, "info", f"Closed {trade.side} {trade.symbol} ({trade.reason.value}) PnL {trade.pnl:.2f}")

    if store is not None and hasattr(store, "record_equity"):
        try:
            store.record_equity(prices)
        except Exception:
            logger.debug("Could not record equity", exc_info=True)

    report.finished_at = utcnow()
    _write_heartbeat(settings.data_dir)
    logger.info(
        "cycle complete | signals=%d decisions=%d orders=%d closed=%d errors=%d",
        len(report.signals),
        len(report.decisions),
        len(report.orders),
        len(report.closed_trades),
        len(report.errors),
    )
    report.prices = prices
    return report
