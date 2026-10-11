"""Execution agent: paper simulator or live CCXT orders.

``PaperExchange`` fills at the supplied price plus ``SLIPPAGE_PCT``, charges
``FEE_RATE``, persists the account to SQLite, and triggers stop-loss and
take-profit on later price checks. It also looks at the high/low of candles
that opened after the position, so a wick through the stop between two checks
is not silently ignored. When a bar touched both the stop and the target the
stop is assumed to have filled first.

``LiveExchange`` sends real orders only when paper mode is off and live
trading was confirmed. The rules it follows:

* **Idempotent orders.** Every order carries a ``clientOrderId``. After a
  timeout the bot looks the order up by that id before retrying, so a slow
  exchange cannot cause a double entry.
* **Stops live on the exchange.** After an entry fills, a separate stop order
  is placed at the planned distance from the *actual* fill price. Its id is
  stored. On swap a reduce-only take-profit is placed too. On spot the
  take-profit is handled by the bot, because two sell orders would both lock
  the same coins and the second would be rejected.
* **Cancel before close.** Any close cancels the stored protective orders
  first, so no orphan stop is left to hit a future position.
* **Fills, not guesses.** When an exchange stop fires, the realized price is
  read from that stop order, not from the current market price.
* **Backstops.** If the stop could not be placed, a software stop runs every
  cycle. If the price stays beyond a protected stop for two checks in a row,
  the bot closes at market anyway.
* **Startup reconciliation** compares local positions with the exchange,
  re-places missing stops and cancels orphan orders with the bot's prefix.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import timedelta
from typing import Any

import ccxt

from config.settings import Settings, assert_trading_mode
from core.errors import InsufficientFundsError, InvalidBracketError, LiveTradingNotConfirmedError
from core.models import (
    Bar,
    CloseReason,
    ClosedTrade,
    MarkResult,
    OrderResult,
    PendingOrder,
    Position,
    RiskAction,
    RiskDecision,
)
from core.redact import redact
from core.retry import call_with_retries
from core.state import PortfolioStore

logger = logging.getLogger("trading.execution")

_RETRYABLE = (ccxt.NetworkError,)
_FATAL = (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.AuthenticationError)
_TERMINAL_STATUS = {"closed", "canceled", "cancelled", "rejected", "expired"}
_CANCELLED = {"canceled", "cancelled", "rejected", "expired"}
# Stop-limit fallback: how far past the trigger the limit price sits.
_STOP_LIMIT_BUFFER = 0.01


# --------------------------------------------------------------------- helpers


def _skipped(decision: RiskDecision) -> OrderResult:
    return OrderResult(
        success=True,
        symbol=decision.symbol,
        side="none",
        amount=0.0,
        price=decision.entry,
        status="skipped",
        message="No action.",
    )


def _failed(decision: RiskDecision, side: str, message: str, *, status: str = "rejected") -> OrderResult:
    message = redact(message)
    logger.warning("%s %s rejected: %s", decision.symbol, side, message)
    return OrderResult(
        success=False,
        symbol=decision.symbol,
        side=side,
        amount=decision.position_size,
        price=decision.entry,
        status=status,
        message=message,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
    )


def _bracket_ok(side: str, entry: float, stop_loss: float, take_profit: float) -> bool:
    if side in {"buy", "long"}:
        return stop_loss < entry < take_profit
    return take_profit < entry < stop_loss


def _shift_bracket(planned_entry: float, fill: float, stop_loss: float, take_profit: float) -> tuple[float, float]:
    """Keep the planned stop and target *distances*, measured from the real fill."""
    if planned_entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        return stop_loss, take_profit
    delta = fill - planned_entry
    return stop_loss + delta, take_profit + delta


def _slipped(price: float, side: str, slippage: float) -> float:
    """Adverse slippage: buys pay more, sells receive less."""
    if side == "buy":
        return price * (1.0 + slippage)
    return price * (1.0 - slippage)


def _exit_side(position: Position) -> str:
    return "sell" if position.side == "long" else "buy"


def _long_stop_hit(position: Position, price: float) -> bool:
    return position.side == "long" and price <= position.stop_loss


def _long_target_hit(position: Position, price: float) -> bool:
    return position.side == "long" and price >= position.take_profit


def _short_stop_hit(position: Position, price: float) -> bool:
    return position.side == "short" and price >= position.stop_loss


def _short_target_hit(position: Position, price: float) -> bool:
    return position.side == "short" and price <= position.take_profit


def _stop_hit(position: Position, price: float) -> bool:
    return _long_stop_hit(position, price) or _short_stop_hit(position, price)


def _target_hit(position: Position, price: float) -> bool:
    return _long_target_hit(position, price) or _short_target_hit(position, price)


def _stop_exit_price(position: Position, price: float) -> float:
    # A gap through the stop fills at the worse observed price, not at the trigger.
    if position.side == "long":
        return min(price, position.stop_loss)
    return max(price, position.stop_loss)


def _bars_after(position: Position, bars: tuple[Bar, ...]) -> list[Bar]:
    """Bars that opened at or after the entry. Earlier bars may hold pre-entry wicks."""
    opened_ms = int(position.opened_at.timestamp() * 1000)
    return [bar for bar in bars if bar.open_ms >= opened_ms]


def _marketable(side: str, limit_price: float, market_price: float) -> bool:
    if side == "buy":
        return market_price <= limit_price
    return market_price >= limit_price


def _average_price(order: dict[str, Any], fallback: float) -> float:
    for key in ("average", "price"):
        value = order.get(key)
        if value:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return fallback


def _currency_bucket(balance: dict[str, Any], currency: str) -> dict[str, Any]:
    bucket = balance.get(currency)
    if isinstance(bucket, dict):
        return bucket
    return {}


def _currency_value(balance: dict[str, Any], currency: str, key: str) -> float:
    bucket = _currency_bucket(balance, currency)
    if key in bucket and bucket[key] is not None:
        return float(bucket[key])
    mapping = balance.get(key)
    if isinstance(mapping, dict) and mapping.get(currency) is not None:
        return float(mapping[currency])
    return 0.0


def _currency_free(balance: dict[str, Any], currency: str) -> float:
    return _currency_value(balance, currency, "free")


def _currency_total(balance: dict[str, Any], currency: str) -> float:
    return _currency_value(balance, currency, "total")


def _infer_reason(position: Position, price: float) -> CloseReason:
    if _stop_hit(position, price):
        return CloseReason.STOP_LOSS
    if _target_hit(position, price):
        return CloseReason.TAKE_PROFIT
    return CloseReason.EXTERNAL


def _looks_like_duplicate(exc: Exception) -> bool:
    text = str(exc).lower()
    return "duplicate" in text or "already exists" in text or "orderlinkid" in text and "exist" in text


# ----------------------------------------------------------------------- paper


class PaperExchange:
    """Virtual exchange. Market fills use the latest price plus slippage and charge ``fee_rate``."""

    def __init__(self, settings: Settings, store: PortfolioStore) -> None:
        self.settings = settings
        self.store = store

    async def execute(self, decision: RiskDecision, market_price: float | None = None) -> OrderResult:
        if decision.action == RiskAction.NO_ACTION:
            return _skipped(decision)
        price_now = market_price if market_price is not None else decision.entry
        position = self.store.get_position(decision.symbol)
        if decision.action == RiskAction.EXECUTE_SELL and position is not None and position.side == "long":
            return self._close(position, price_now, CloseReason.SIGNAL)
        if decision.action == RiskAction.EXECUTE_BUY and position is not None and position.side == "short":
            return self._close(position, price_now, CloseReason.SIGNAL)
        if position is not None:
            return _failed(decision, position.side, "Position already open.")
        side = "buy" if decision.action == RiskAction.EXECUTE_BUY else "sell"
        return self._open(decision, side, price_now)

    def _open(self, decision: RiskDecision, side: str, market_price: float) -> OrderResult:
        if decision.position_size <= 0 or market_price <= 0:
            return _failed(decision, side, "Size or price is not positive.")
        is_limit = self.settings.order_type == "limit"
        limit_price = decision.entry if is_limit else market_price
        if is_limit and not _marketable(side, limit_price, market_price):
            pending = self.store.add_pending(
                symbol=decision.symbol,
                side=side,
                size=decision.position_size,
                limit_price=limit_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
            )
            logger.info("Paper limit resting %s %s @ %s", side, decision.symbol, limit_price)
            return OrderResult(
                success=True,
                symbol=decision.symbol,
                side=side,
                amount=decision.position_size,
                price=limit_price,
                status="open",
                message="Paper limit order is resting.",
                order_id=pending.id,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                opened=False,
            )
        if is_limit:
            fill_price = min(limit_price, market_price) if side == "buy" else max(limit_price, market_price)
        else:
            fill_price = _slipped(market_price, side, self.settings.slippage_pct)
        stop_loss, take_profit = _shift_bracket(decision.entry, fill_price, decision.stop_loss, decision.take_profit)
        if not _bracket_ok(side, fill_price, stop_loss, take_profit):
            return _failed(
                decision,
                side,
                f"Bracket is invalid at fill {fill_price:.8g} (stop {stop_loss:.8g}, target {take_profit:.8g}).",
            )
        try:
            position = self.store.open_position(
                symbol=decision.symbol,
                side="long" if side == "buy" else "short",
                size=decision.position_size,
                price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exchange_protected=True,
            )
        except InsufficientFundsError as exc:
            return _failed(decision, side, f"Insufficient funds: {exc}", status="insufficient_funds")
        logger.info(
            "Paper fill %s %s size=%s price=%s sl=%s tp=%s",
            side,
            decision.symbol,
            position.size,
            fill_price,
            stop_loss,
            take_profit,
        )
        return OrderResult(
            success=True,
            symbol=decision.symbol,
            side=side,
            amount=position.size,
            price=fill_price,
            status="closed",
            message="Paper order filled.",
            order_id=position.order_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened=True,
        )

    def _close(self, position: Position, price: float, reason: CloseReason, *, slip: bool = True) -> OrderResult:
        side = _exit_side(position)
        if price <= 0:
            return OrderResult(
                success=False,
                symbol=position.symbol,
                side=side,
                amount=position.size,
                price=price,
                status="rejected",
                message="Cannot close without a positive price.",
            )
        fill = _slipped(price, side, self.settings.slippage_pct) if slip else price
        trade = self.store.close_position(position.symbol, fill, reason)
        self._maybe_cooldown(trade)
        logger.info(
            "Paper close %s %s price=%s pnl=%.4f reason=%s",
            position.symbol,
            position.side,
            fill,
            trade.pnl,
            reason.value,
        )
        return OrderResult(
            success=True,
            symbol=position.symbol,
            side=side,
            amount=position.size,
            price=fill,
            status="closed",
            message=f"Paper position closed ({reason.value}).",
            opened=False,
            closed_trade=trade,
        )

    async def mark_prices(
        self,
        prices: dict[str, float],
        bars: dict[str, tuple[Bar, ...]] | None = None,
    ) -> MarkResult:
        result = MarkResult()
        bars = bars or {}
        now = self.store.now()
        ttl = timedelta(minutes=self.settings.pending_order_ttl_minutes)
        for pending in list(self.store.pending_orders()):
            price = prices.get(pending.symbol)
            if price is None:
                continue
            if _marketable(pending.side, pending.limit_price, price):
                fill_price = min(pending.limit_price, price) if pending.side == "buy" else max(pending.limit_price, price)
                self.store.remove_pending(pending.id)
                opened = self._fill_pending(pending, fill_price)
                if opened.success and opened.opened:
                    result.opened.append(opened)
                elif not opened.success:
                    result.errors.append(opened.message)
            elif pending.created_at + ttl <= now:
                self.store.remove_pending(pending.id)
                result.errors.append(
                    f"{pending.symbol}: paper limit {pending.side} @ {pending.limit_price:.8g} expired unfilled."
                )
        for position in list(self.store.view(prices).positions.values()):
            price = prices.get(position.symbol)
            if price is None:
                continue
            closed = self._check_exit(position, price, bars.get(position.symbol, ()))
            if closed is None:
                continue
            if closed.closed_trade is not None:
                result.closed.append(closed.closed_trade)
            elif not closed.success:
                result.errors.append(closed.message)
        return result

    def _check_exit(self, position: Position, price: float, bars: tuple[Bar, ...]) -> OrderResult | None:
        later = _bars_after(position, bars)
        if position.side == "long":
            wick_stop = any(bar.low <= position.stop_loss for bar in later)
            wick_target = any(bar.high >= position.take_profit for bar in later)
        else:
            wick_stop = any(bar.high >= position.stop_loss for bar in later)
            wick_target = any(bar.low <= position.take_profit for bar in later)
        if _stop_hit(position, price) or wick_stop:
            # Conservative: when the stop and the target were both touched, assume the stop came first.
            fill = _stop_exit_price(position, price) if _stop_hit(position, price) else position.stop_loss
            return self._close(position, fill, CloseReason.STOP_LOSS)
        if _target_hit(position, price) or wick_target:
            # Targets are limits: fill at the target, not through it, and without slippage.
            return self._close(position, position.take_profit, CloseReason.TAKE_PROFIT, slip=False)
        return None

    def _fill_pending(self, pending: PendingOrder, fill_price: float) -> OrderResult:
        decision = RiskDecision(
            action=RiskAction.EXECUTE_BUY if pending.side == "buy" else RiskAction.EXECUTE_SELL,
            symbol=pending.symbol,
            position_size=pending.size,
            entry=fill_price,
            stop_loss=pending.stop_loss,
            take_profit=pending.take_profit,
            confidence=0.0,
            reasoning="Resting paper limit filled.",
        )
        if not _bracket_ok(pending.side, fill_price, pending.stop_loss, pending.take_profit):
            return _failed(decision, pending.side, "Filled limit crossed the protective bracket.")
        try:
            position = self.store.open_position(
                symbol=pending.symbol,
                side="long" if pending.side == "buy" else "short",
                size=pending.size,
                price=fill_price,
                stop_loss=pending.stop_loss,
                take_profit=pending.take_profit,
                order_id=pending.id,
                exchange_protected=True,
            )
        except InsufficientFundsError as exc:
            return _failed(decision, pending.side, f"Insufficient funds: {exc}", status="insufficient_funds")
        return OrderResult(
            success=True,
            symbol=pending.symbol,
            side=pending.side,
            amount=position.size,
            price=fill_price,
            status="closed",
            message="Paper limit filled.",
            order_id=pending.id,
            stop_loss=pending.stop_loss,
            take_profit=pending.take_profit,
            opened=True,
        )

    async def close_all(self, prices: dict[str, float], reason: CloseReason = CloseReason.MANUAL) -> list[OrderResult]:
        results = []
        for pending in self.store.pending_orders():
            self.store.remove_pending(pending.id)
        for position in self.store.positions():
            price = prices.get(position.symbol) or position.entry
            results.append(self._close(position, price, reason))
        return results

    async def reconcile(self, symbols: list[str]) -> list[str]:
        del symbols
        return []

    def _maybe_cooldown(self, trade: ClosedTrade) -> None:
        if trade.reason != CloseReason.STOP_LOSS:
            return
        until = trade.closed_at + timedelta(minutes=self.settings.cooldown_minutes)
        self.store.set_cooldown(trade.symbol, until, "stop_loss")

    async def close(self) -> None:
        return None


# ------------------------------------------------------------------------ live


class LiveExchange:
    """Authenticated CCXT execution. Refuses to construct unless live mode is confirmed."""

    def __init__(self, settings: Settings, exchange: Any, store: PortfolioStore) -> None:
        if settings.paper_trading or not settings.live_trading_confirm:
            raise LiveTradingNotConfirmedError(
                "LiveExchange requires PAPER_TRADING=false and LIVE_TRADING_CONFIRM=true."
            )
        assert_trading_mode(settings)
        self.settings = settings
        self.exchange = exchange
        self.store = store
        self._markets_loaded = False
        # Symbols whose price was beyond a protected stop at the previous check.
        self._breach_seen: set[str] = set()

    # ------------------------------------------------------------ entry path

    async def execute(self, decision: RiskDecision, market_price: float | None = None) -> OrderResult:
        if decision.action == RiskAction.NO_ACTION:
            return _skipped(decision)
        position = self.store.get_position(decision.symbol)
        if decision.action == RiskAction.EXECUTE_SELL and position is not None and position.side == "long":
            return await self._close_live(position, market_price or decision.entry, CloseReason.SIGNAL)
        if decision.action == RiskAction.EXECUTE_BUY and position is not None and position.side == "short":
            return await self._close_live(position, market_price or decision.entry, CloseReason.SIGNAL)
        if position is not None:
            return _failed(decision, position.side, "Position already open.")
        side = "buy" if decision.action == RiskAction.EXECUTE_BUY else "sell"
        return await self._open_live(decision, side)

    async def _open_live(self, decision: RiskDecision, side: str) -> OrderResult:
        if decision.position_size <= 0:
            return _failed(decision, side, "Size is not positive.")
        try:
            await self._ensure_markets()
            amount = self._amount(decision.symbol, decision.position_size)
        except Exception as exc:
            return _failed(decision, side, f"Could not prepare the order: {exc}")
        if amount <= 0:
            return _failed(decision, side, "Amount rounded to zero at exchange precision.")
        price = self._price(decision.symbol, decision.entry) if self.settings.order_type == "limit" else None
        try:
            order = await self._create_idempotent(
                decision.symbol, self.settings.order_type, side, amount, price, {}
            )
        except ccxt.InsufficientFunds as exc:
            return _failed(decision, side, f"Insufficient funds: {exc}", status="insufficient_funds")
        except ccxt.AuthenticationError as exc:
            return _failed(decision, side, f"Authentication failed: {exc}")
        except ccxt.NetworkError as exc:
            return _failed(decision, side, f"Network error after retries (order not found by client id): {exc}")
        except ccxt.ExchangeError as exc:
            return _failed(decision, side, f"Exchange rejected the order: {exc}")
        except Exception as exc:
            return _failed(decision, side, f"Order failed: {exc}")

        resolved = await self._resolve_fill(order, decision.symbol)
        filled = float(resolved.get("filled") or 0)
        status = str(resolved.get("status") or "open")
        order_id = str(resolved.get("id") or "")
        if filled <= 0:
            if order_id and status == "open":
                # Keep the exchange id so the next cycle adopts the fill instead of re-entering.
                self.store.add_pending(
                    symbol=decision.symbol,
                    side=side,
                    size=amount,
                    limit_price=decision.entry,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    order_id=order_id,
                    client_order_id=resolved.get("clientOrderId"),
                )
                return OrderResult(
                    success=True,
                    symbol=decision.symbol,
                    side=side,
                    amount=amount,
                    price=decision.entry,
                    status="open",
                    message="Order is resting; it will be adopted when it fills.",
                    order_id=order_id,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                )
            return _failed(decision, side, f"Order not filled (status {status}).", status=status)
        if status == "open" and order_id:
            # Partial fill on an order that is still working: take what filled, cancel the rest.
            await self._cancel_quietly(order_id, decision.symbol)
        average = _average_price(resolved, decision.entry)
        return await self._adopt_fill(
            symbol=decision.symbol,
            side=side,
            filled=filled,
            average=average,
            planned_entry=decision.entry,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            order_id=order_id or None,
        )

    async def _adopt_fill(
        self,
        *,
        symbol: str,
        side: str,
        filled: float,
        average: float,
        planned_entry: float,
        stop_loss: float,
        take_profit: float,
        order_id: str | None,
    ) -> OrderResult:
        """Record a filled entry, then protect it on the exchange."""
        stop_loss, take_profit = _shift_bracket(planned_entry, average, stop_loss, take_profit)
        position = self.store.open_position(
            symbol=symbol,
            side="long" if side == "buy" else "short",
            size=filled,
            price=average,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=order_id,
            exchange_protected=False,
            enforce_cash=False,  # the exchange already filled; never lose track of it
        )
        sl_id, tp_id = await self._protect(position)
        self.store.update_protection(symbol, exchange_protected=sl_id is not None, sl_order_id=sl_id, tp_order_id=tp_id)
        message = "Live order filled."
        if sl_id is None:
            message += " WARNING: the exchange did not accept a stop. A software stop runs every cycle."
            logger.error("Entry filled for %s but no exchange stop was accepted.", symbol)
        return OrderResult(
            success=True,
            symbol=symbol,
            side=side,
            amount=filled,
            price=average,
            status="closed",
            message=message,
            order_id=order_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened=True,
        )

    # ------------------------------------------------------------ protection

    async def _protect(self, position: Position, *, include_tp: bool = True) -> tuple[str | None, str | None]:
        """Place the stop (and on swap the take-profit). Returns their exchange ids."""
        symbol = position.symbol
        close_side = _exit_side(position)
        reduce_only = self.settings.market_type == "swap"
        extra: dict[str, Any] = {"reduceOnly": True} if reduce_only else {}
        amount = await self._closable_amount(position)
        if amount <= 0:
            return None, None
        stop = self._price(symbol, position.stop_loss)
        buffer = -_STOP_LIMIT_BUFFER if close_side == "sell" else _STOP_LIMIT_BUFFER
        stop_limit = self._price(symbol, position.stop_loss * (1 + buffer))
        sl_id = await self._first_accepted(
            symbol,
            close_side,
            amount,
            [
                # CCXT's unified stop-loss order.
                ("market", None, {"stopLossPrice": stop, **extra}),
                # Generic trigger order (Bybit and others).
                ("market", None, {"triggerPrice": stop, **extra}),
                # Venues that only allow stop-limit on spot (for example Binance STOP_LOSS_LIMIT).
                ("limit", stop_limit, {"stopLossPrice": stop, **extra}),
            ],
        )
        tp_id = None
        if reduce_only and include_tp:
            target = self._price(symbol, position.take_profit)
            tp_id = await self._first_accepted(
                symbol,
                close_side,
                amount,
                [
                    ("market", None, {"takeProfitPrice": target, **extra}),
                    ("limit", target, {**extra}),
                ],
            )
        return sl_id, tp_id

    async def _first_accepted(
        self,
        symbol: str,
        side: str,
        amount: float,
        attempts: list[tuple[str, float | None, dict[str, Any]]],
    ) -> str | None:
        for order_type, price, params in attempts:
            try:
                order = await self._create_idempotent(symbol, order_type, side, amount, price, params)
            except ccxt.InsufficientFunds as exc:
                logger.warning("Protection order for %s lacks funds: %s", symbol, redact(exc))
                return None
            except ccxt.NetworkError as exc:
                logger.warning("Network error placing protection for %s: %s", symbol, redact(exc))
                return None
            except ccxt.ExchangeError as exc:
                logger.info("Protection %s %s rejected for %s: %s", order_type, params, symbol, redact(exc))
                continue
            order_id = order.get("id") if isinstance(order, dict) else None
            if order_id:
                logger.info("Protection order accepted %s %s %s id=%s", order_type, side, symbol, order_id)
                return str(order_id)
        return None

    async def _cancel_quietly(self, order_id: str | None, symbol: str) -> bool:
        """Cancel one order. ``True`` when it is gone (cancelled, filled or unknown)."""
        if not order_id:
            return True
        try:
            await self._retry(lambda: self.exchange.cancel_order(order_id, symbol))
            return True
        except ccxt.OrderNotFound:
            return True
        except Exception as exc:
            logger.warning("Could not cancel %s order %s: %s", symbol, order_id, redact(exc))
            return False

    async def _cancel_protection(self, position: Position) -> bool:
        sl_ok = await self._cancel_quietly(position.sl_order_id, position.symbol)
        tp_ok = await self._cancel_quietly(position.tp_order_id, position.symbol)
        return sl_ok and tp_ok

    # --------------------------------------------------------------- closing

    async def _close_live(self, position: Position, price: float, reason: CloseReason) -> OrderResult:
        side = _exit_side(position)
        params: dict[str, Any] = {}
        if self.settings.market_type != "spot":
            params["reduceOnly"] = True

        def failure(status: str, message: str) -> OrderResult:
            return OrderResult(
                success=False,
                symbol=position.symbol,
                side=side,
                amount=position.size,
                price=price,
                status=status,
                message=redact(message),
            )

        # Free the coins (spot) and remove orphans-to-be before the market close.
        if not await self._cancel_protection(position):
            logger.warning("Closing %s although a protective order could not be cancelled.", position.symbol)
        try:
            await self._ensure_markets()
            amount = await self._closable_amount(position)
            if amount <= 0:
                trade = self.store.close_position(position.symbol, price, CloseReason.EXTERNAL)
                return OrderResult(
                    success=True,
                    symbol=position.symbol,
                    side=side,
                    amount=0.0,
                    price=price,
                    status="closed",
                    message="Nothing left to sell on the exchange; closed locally as external.",
                    closed_trade=trade,
                )
            order = await self._create_idempotent(position.symbol, "market", side, amount, None, params)
            resolved = await self._resolve_fill(order, position.symbol)
        except ccxt.InsufficientFunds as exc:
            return failure("insufficient_funds", f"Insufficient funds while closing: {exc}")
        except ccxt.NetworkError as exc:
            return failure("rejected", f"Network error while closing: {exc}")
        except ccxt.ExchangeError as exc:
            return failure("rejected", f"Exchange rejected the close: {exc}")
        exit_price = _average_price(resolved, price)
        trade = self.store.close_position(position.symbol, exit_price, reason)
        self._breach_seen.discard(position.symbol)
        self._maybe_cooldown(trade)
        return OrderResult(
            success=True,
            symbol=position.symbol,
            side=side,
            amount=amount,
            price=exit_price,
            status="closed",
            message=f"Live position closed ({reason.value}).",
            order_id=str(resolved.get("id") or "") or None,
            closed_trade=trade,
        )

    async def _closable_amount(self, position: Position) -> float:
        """On spot, never try to sell more coin than the account holds.

        Exchanges often take the buy fee in the base coin, so a 0.1 BTC buy
        can leave 0.0999 BTC. Selling the recorded 0.1 would be rejected.
        """
        size = position.size
        if self.settings.market_type == "spot" and position.side == "long":
            balance = await self._retry(lambda: self.exchange.fetch_balance())
            base = position.symbol.split("/")[0]
            held = _currency_total(balance, base)
            if held > 0:
                size = min(size, held)
        return self._amount(position.symbol, size)

    async def close_all(self, prices: dict[str, float], reason: CloseReason = CloseReason.MANUAL) -> list[OrderResult]:
        results: list[OrderResult] = []
        for pending in self.store.pending_orders():
            if await self._cancel_quietly(pending.id, pending.symbol):
                self.store.remove_pending(pending.id)
        for position in self.store.positions():
            price = prices.get(position.symbol) or position.entry
            results.append(await self._close_live(position, price, reason))
        return results

    # ------------------------------------------------------- per-cycle checks

    async def mark_prices(
        self,
        prices: dict[str, float],
        bars: dict[str, tuple[Bar, ...]] | None = None,
    ) -> MarkResult:
        del bars  # live stops are real orders; candles are not needed here
        result = MarkResult()
        await self._poll_pending(result)
        for position in list(self.store.positions()):
            try:
                await self._check_position(position, prices.get(position.symbol), result)
            except Exception as exc:
                result.errors.append(redact(f"Position check failed for {position.symbol}: {exc}"))
        try:
            await self._sync_cash(prices)
        except Exception as exc:
            result.errors.append(redact(f"Balance sync failed: {exc}"))
            logger.warning("Balance sync failed: %s", redact(exc))
        return result

    async def _poll_pending(self, result: MarkResult) -> None:
        now = self.store.now()
        ttl = timedelta(minutes=self.settings.pending_order_ttl_minutes)
        for pending in list(self.store.pending_orders()):
            try:
                order = await self._retry(lambda pending=pending: self.exchange.fetch_order(pending.id, pending.symbol))
            except Exception as exc:
                result.errors.append(redact(f"Could not poll {pending.symbol} order {pending.id}: {exc}"))
                continue
            status = str(order.get("status") or "")
            filled = float(order.get("filled") or 0)
            expired = pending.created_at + ttl <= now
            if status == "open" and expired:
                if not await self._cancel_quietly(pending.id, pending.symbol):
                    result.errors.append(f"{pending.symbol}: could not cancel the expired entry {pending.id}.")
                    continue
                try:
                    order = await self._retry(lambda pending=pending: self.exchange.fetch_order(pending.id, pending.symbol))
                    filled = float(order.get("filled") or 0)
                except Exception:
                    pass
                status = "canceled"
                if filled <= 0:
                    result.errors.append(f"{pending.symbol}: limit entry expired unfilled and was cancelled.")
            if status == "open":
                continue
            self.store.remove_pending(pending.id)
            if filled <= 0:
                continue
            opened = await self._adopt_fill(
                symbol=pending.symbol,
                side=pending.side,
                filled=filled,
                average=_average_price(order, pending.limit_price),
                planned_entry=pending.limit_price,
                stop_loss=pending.stop_loss,
                take_profit=pending.take_profit,
                order_id=pending.id,
            )
            result.opened.append(opened)

    async def _order_status(self, order_id: str | None, symbol: str) -> dict[str, Any] | None:
        if not order_id:
            return None
        try:
            return await self._retry(lambda: self.exchange.fetch_order(order_id, symbol))
        except ccxt.OrderNotFound:
            return {"id": order_id, "status": "missing", "filled": 0}

    async def _check_position(self, position: Position, price: float | None, result: MarkResult) -> None:
        symbol = position.symbol
        # 1. Did one of our protective orders fill on the exchange?
        stop_state = ""
        for order_id, reason, sibling in (
            (position.sl_order_id, CloseReason.STOP_LOSS, position.tp_order_id),
            (position.tp_order_id, CloseReason.TAKE_PROFIT, position.sl_order_id),
        ):
            order = await self._order_status(order_id, symbol)
            if order is None:
                continue
            status = str(order.get("status") or "")
            if reason == CloseReason.STOP_LOSS:
                stop_state = status
            if status in {"closed", "filled"} and float(order.get("filled") or 0) > 0:
                await self._cancel_quietly(sibling, symbol)
                fallback = position.stop_loss if reason == CloseReason.STOP_LOSS else position.take_profit
                trade = self.store.close_position(symbol, _average_price(order, fallback), reason)
                self._breach_seen.discard(symbol)
                self._maybe_cooldown(trade)
                result.closed.append(trade)
                return

        # 2. Flat on the exchange without an order we can identify.
        if await self._exchange_flat(position):
            mark = price or position.entry
            await self._cancel_protection(position)
            trade = self.store.close_position(symbol, mark, _infer_reason(position, mark))
            self._maybe_cooldown(trade)
            result.closed.append(trade)
            result.errors.append(
                f"{symbol}: position is flat on the exchange but no bot order explains it. "
                f"Recorded at {mark:.8g}; check the exchange trade history."
            )
            return

        # 3. Our stop vanished (cancelled by hand, expired, rejected). Put it back.
        if position.sl_order_id and stop_state in _CANCELLED | {"missing"}:
            result.errors.append(f"{symbol}: exchange stop {position.sl_order_id} is {stop_state}. Re-placing it.")
            sl_id, _ = await self._protect(position, include_tp=False)
            self.store.update_protection(
                symbol, exchange_protected=sl_id is not None, sl_order_id=sl_id, tp_order_id=position.tp_order_id
            )
            position.sl_order_id = sl_id
            position.exchange_protected = sl_id is not None

        if price is None:
            return

        # 4. Spot take-profit is managed by the bot (no exchange TP order exists).
        if position.tp_order_id is None and _target_hit(position, price):
            closed = await self._close_live(position, position.take_profit, CloseReason.TAKE_PROFIT)
            self._collect(closed, result)
            return

        # 5. Software stop, and the backstop for a protected stop that did not fire.
        if _stop_hit(position, price):
            if not position.exchange_protected or symbol in self._breach_seen:
                if position.exchange_protected:
                    result.errors.append(
                        f"{symbol}: price stayed beyond the exchange stop for two checks. Closing at market."
                    )
                closed = await self._close_live(position, _stop_exit_price(position, price), CloseReason.STOP_LOSS)
                self._collect(closed, result)
                return
            self._breach_seen.add(symbol)
            result.errors.append(
                f"{symbol}: price {price:.8g} is beyond the stop; giving the exchange stop one more check."
            )
        else:
            self._breach_seen.discard(symbol)

    @staticmethod
    def _collect(closed: OrderResult, result: MarkResult) -> None:
        if closed.closed_trade is not None:
            result.closed.append(closed.closed_trade)
        elif not closed.success:
            result.errors.append(closed.message)

    async def _exchange_flat(self, position: Position) -> bool:
        if self.settings.market_type == "swap" and hasattr(self.exchange, "fetch_positions"):
            rows = await self._retry(lambda: self.exchange.fetch_positions([position.symbol]))
            for row in rows or []:
                contracts = float(row.get("contracts") or 0)
                if contracts > 0:
                    return False
            return True
        balance = await self._retry(lambda: self.exchange.fetch_balance())
        base = position.symbol.split("/")[0]
        total = _currency_total(balance, base)
        return total < position.size * 0.5

    async def _sync_cash(self, marks: dict[str, float]) -> None:
        balance = await self._retry(lambda: self.exchange.fetch_balance())
        free = _currency_free(balance, self.settings.quote_currency)
        self.store.sync_quote_balance(free, marks)

    # --------------------------------------------------------- reconciliation

    async def reconcile(self, symbols: list[str]) -> list[str]:
        """Compare local state with the exchange. Returns human-readable findings."""
        notes: list[str] = []
        await self._ensure_markets()
        if self.settings.market_type == "swap":
            for symbol in symbols:
                for method, args in (
                    ("set_margin_mode", ("isolated", symbol)),
                    ("set_leverage", (self.settings.leverage, symbol)),
                ):
                    fn = getattr(self.exchange, method, None)
                    if fn is None:
                        continue
                    try:
                        await fn(*args)
                    except Exception as exc:  # already set, or not supported
                        logger.info("%s %s: %s", method, symbol, redact(exc))
        known: set[str] = set()
        for position in self.store.positions():
            known.update(i for i in (position.sl_order_id, position.tp_order_id, position.order_id) if i)
            result = MarkResult()
            try:
                await self._check_position(position, None, result)
            except Exception as exc:
                notes.append(redact(f"{position.symbol}: reconciliation failed: {exc}"))
                continue
            notes.extend(result.errors)
            notes.extend(
                f"{trade.symbol}: closed while the bot was offline ({trade.reason.value}, PnL {trade.pnl:.2f})"
                for trade in result.closed
            )
            current = self.store.get_position(position.symbol)
            if current is not None:
                known.update(i for i in (current.sl_order_id, current.tp_order_id) if i)
            if current is not None and current.sl_order_id is None:
                sl_id, tp_id = await self._protect(current)
                self.store.update_protection(
                    current.symbol, exchange_protected=sl_id is not None, sl_order_id=sl_id, tp_order_id=tp_id
                )
                notes.append(
                    f"{current.symbol}: had no exchange stop; "
                    + ("placed one." if sl_id else "could not place one, software stop active.")
                )
                known.update(i for i in (sl_id, tp_id) if i)
        known.update(order.id for order in self.store.pending_orders())
        fetch_open = getattr(self.exchange, "fetch_open_orders", None)
        if fetch_open is not None:
            prefix = self.settings.client_order_prefix
            for symbol in symbols:
                try:
                    orders = await self._retry(lambda symbol=symbol: fetch_open(symbol))
                except Exception as exc:
                    notes.append(redact(f"{symbol}: could not list open orders: {exc}"))
                    continue
                for order in orders or []:
                    client_id = str(order.get("clientOrderId") or "")
                    order_id = str(order.get("id") or "")
                    if client_id.startswith(prefix) and order_id not in known:
                        if await self._cancel_quietly(order_id, symbol):
                            notes.append(f"{symbol}: cancelled orphan bot order {order_id} ({order.get('type')}).")
        return notes

    # ------------------------------------------------------------- plumbing

    def _client_id(self) -> str:
        return f"{self.settings.client_order_prefix}{uuid.uuid4().hex[:24]}"

    async def _find_by_client_id(self, symbol: str, client_id: str) -> dict[str, Any] | None:
        for method in ("fetch_open_orders", "fetch_closed_orders"):
            fn = getattr(self.exchange, method, None)
            if fn is None:
                continue
            try:
                orders = await fn(symbol)
            except Exception as exc:
                logger.info("%s lookup failed for %s: %s", method, symbol, redact(exc))
                continue
            for order in orders or []:
                if str(order.get("clientOrderId") or "") == client_id:
                    return order
        return None

    async def _create_idempotent(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Create an order exactly once, even if the response is lost.

        A timeout does not mean the order was rejected. Before every retry the
        order is looked up by its ``clientOrderId``; the same id is reused so
        the exchange itself rejects a duplicate.
        """
        client_id = self._client_id()
        full_params = {**params, "clientOrderId": client_id}
        delay = self.settings.retry_base_delay
        last: BaseException | None = None
        for attempt in range(1, self.settings.retry_attempts + 1):
            try:
                return await self.exchange.create_order(symbol, order_type, side, amount, price, full_params)
            except _FATAL as exc:
                if attempt > 1 and _looks_like_duplicate(exc):
                    found = await self._find_by_client_id(symbol, client_id)
                    if found is not None:
                        return found
                raise
            except _RETRYABLE as exc:
                last = exc
                found = await self._find_by_client_id(symbol, client_id)
                if found is not None:
                    logger.warning("Order %s for %s was accepted despite a network error.", client_id, symbol)
                    return found
                if attempt >= self.settings.retry_attempts:
                    break
                await asyncio.sleep(delay)
                delay *= 2
        assert last is not None
        raise last

    async def _resolve_fill(self, order: dict[str, Any], symbol: str) -> dict[str, Any]:
        status = str(order.get("status") or "")
        filled = float(order.get("filled") or 0)
        if status in _TERMINAL_STATUS or (filled > 0 and status not in {"open", ""}):
            return order
        order_id = order.get("id")
        if not order_id:
            return order
        last = order
        for _ in range(self.settings.order_poll_attempts):
            await asyncio.sleep(self.settings.order_poll_delay)
            try:
                last = await self._retry(lambda: self.exchange.fetch_order(order_id, symbol))
            except ccxt.ExchangeError as exc:
                logger.warning("Order poll failed for %s: %s", symbol, redact(exc))
                break
            if str(last.get("status") or "") in _TERMINAL_STATUS:
                return last
        return last

    async def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        load = getattr(self.exchange, "load_markets", None)
        if load is not None:
            await self._retry(load)
        self._markets_loaded = True

    def _amount(self, symbol: str, amount: float) -> float:
        precise = getattr(self.exchange, "amount_to_precision", None)
        if precise is None:
            return float(f"{amount:.8f}")
        return float(precise(symbol, amount))

    def _price(self, symbol: str, price: float) -> float:
        precise = getattr(self.exchange, "price_to_precision", None)
        if precise is None:
            return float(price)
        return float(precise(symbol, price))

    async def _retry(self, fn: Any) -> Any:
        """Retry for reads and cancels. Never use this for ``create_order``."""
        return await call_with_retries(
            fn,
            attempts=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay,
            retry_exceptions=_RETRYABLE,
            fatal_exceptions=_FATAL,
        )

    def _maybe_cooldown(self, trade: ClosedTrade) -> None:
        if trade.reason != CloseReason.STOP_LOSS:
            return
        until = trade.closed_at + timedelta(minutes=self.settings.cooldown_minutes)
        self.store.set_cooldown(trade.symbol, until, "stop_loss")

    async def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


# ---------------------------------------------------------------------- facade


class ExecutionAgent:
    """Facade the orchestrator calls. Paper mode never constructs a live client."""

    def __init__(self, settings: Settings, store: PortfolioStore, live_exchange: Any | None = None) -> None:
        self.settings = settings
        self.store = store
        if settings.paper_trading:
            self.backend: PaperExchange | LiveExchange = PaperExchange(settings, store)
        else:
            if live_exchange is None:
                raise LiveTradingNotConfirmedError("Live execution requires an authenticated CCXT client.")
            self.backend = LiveExchange(settings, live_exchange, store)

    async def execute(self, decision: RiskDecision, market_price: float | None = None) -> OrderResult:
        try:
            return await self.backend.execute(decision, market_price)
        except InvalidBracketError as exc:
            return _failed(decision, "unknown", str(exc))

    async def mark_prices(
        self,
        prices: dict[str, float],
        bars: dict[str, tuple[Bar, ...]] | None = None,
    ) -> MarkResult:
        return await self.backend.mark_prices(prices, bars)

    async def close_all(self, prices: dict[str, float]) -> list[OrderResult]:
        return await self.backend.close_all(prices)

    async def reconcile(self, symbols: list[str]) -> list[str]:
        return await self.backend.reconcile(symbols)

    def portfolio_view(self, marks: dict[str, float] | None = None) -> Any:
        return self.store.view(marks)

    async def close(self) -> None:
        await self.backend.close()
        self.store.close()
