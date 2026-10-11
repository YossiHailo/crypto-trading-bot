"""Live execution against a fake exchange: idempotency, protection, closes, reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import ccxt
import pytest

from agents.execution import LiveExchange
from core.errors import LiveTradingNotConfirmedError
from core.exchange import build_ccxt_exchange
from core.models import CloseReason, RiskAction, RiskDecision
from core.state import PortfolioStore
from tests.conftest import make_settings

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
_TRIGGER_KEYS = ("stopLossPrice", "takeProfitPrice", "triggerPrice", "stopPrice")


class FakeExchange:
    """A small but stateful model of a CCXT exchange (spot or swap)."""

    def __init__(self, market_type: str = "spot") -> None:
        self.market_type = market_type
        self.enableRateLimit = True
        self.orders: dict[str, dict] = {}
        self.created: list[dict] = []
        self.cancelled: list[str] = []
        self.calls: list[tuple] = []
        self.market_price = 100.0
        self.base_fee = 0.0  # fee taken in the base coin on buys
        self.fail_before: list[Exception] = []  # raised before the order reaches the book
        self.lose_response = 0  # order is created, but the response is lost
        self.rejected_types: set[str] = set()  # trigger param names the venue refuses
        self.fill_limits = True
        self.contracts = 0.0
        self.balance = {
            "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            "USDT": {"free": 10_000.0, "used": 0.0, "total": 10_000.0},
        }

    # -- precision -------------------------------------------------------
    def amount_to_precision(self, symbol, amount):
        return str(round(float(amount), 6))

    def price_to_precision(self, symbol, price):
        return str(round(float(price), 2))

    async def load_markets(self):
        return {"BTC/USDT": {}}

    # -- balance helpers -------------------------------------------------
    def _btc(self, key):
        return float(self.balance["BTC"][key])

    def _set_btc(self, free=None, used=None):
        b = self.balance["BTC"]
        if free is not None:
            b["free"] = round(free, 10)
        if used is not None:
            b["used"] = round(used, 10)
        b["total"] = round(b["free"] + b["used"], 10)

    def _fill_market(self, side, amount, price):
        if self.market_type == "swap":
            self.contracts += amount if side == "buy" else -amount
            self.contracts = round(self.contracts, 10)
            return
        if side == "buy":
            self._set_btc(free=self._btc("free") + amount * (1 - self.base_fee))
            self.balance["USDT"]["free"] -= amount * price
        else:
            self.balance["USDT"]["free"] += amount * price

    # -- orders ----------------------------------------------------------
    async def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        params = dict(params or {})
        self.calls.append(("create", order_type, side, amount, price, params))
        if self.fail_before:
            raise self.fail_before.pop(0)
        cid = params.get("clientOrderId")
        if cid and any(o.get("clientOrderId") == cid for o in self.orders.values()):
            raise ccxt.InvalidOrder("Duplicate order sent.")
        trigger_key = next((k for k in _TRIGGER_KEYS if k in params), None)
        if trigger_key in self.rejected_types:
            raise ccxt.InvalidOrder(f"unknown parameter {trigger_key}")
        order = {
            "id": str(len(self.orders) + 1),
            "clientOrderId": cid,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": float(amount),
            "price": price,
            "params": params,
            "trigger": trigger_key,
            "status": "open",
            "filled": 0.0,
            "average": None,
        }
        amount = float(amount)
        if trigger_key or (order_type == "limit" and not self.fill_limits):
            if self.market_type == "spot" and side == "sell":
                if self._btc("free") + 1e-12 < amount:
                    raise ccxt.InsufficientFunds("balance locked")
                self._set_btc(free=self._btc("free") - amount, used=self._btc("used") + amount)
        else:
            if self.market_type == "spot" and side == "sell" and self._btc("free") + 1e-12 < amount:
                raise ccxt.InsufficientFunds("Account has insufficient balance for requested action.")
            fill = float(price) if (order_type == "limit" and price) else self.market_price
            self._fill_market(side, amount, fill)
            if self.market_type == "spot" and side == "sell":
                self._set_btc(free=self._btc("free") - amount)
            order.update(status="closed", filled=amount, average=fill)
        self.orders[order["id"]] = order
        self.created.append(order)
        if self.lose_response > 0:
            self.lose_response -= 1
            raise ccxt.RequestTimeout("read timeout")
        return dict(order)

    async def cancel_order(self, order_id, symbol):
        order = self.orders.get(order_id)
        if order is None:
            raise ccxt.OrderNotFound(order_id)
        if order["status"] == "open":
            order["status"] = "canceled"
            self.cancelled.append(order_id)
            if self.market_type == "spot" and order["side"] == "sell":
                self._set_btc(free=self._btc("free") + order["amount"], used=self._btc("used") - order["amount"])
        return dict(order)

    def trigger(self, order_id, price):
        """Simulate the exchange firing a resting stop or target."""
        order = self.orders[order_id]
        order.update(status="closed", filled=order["amount"], average=price)
        if self.market_type == "spot":
            self._set_btc(used=self._btc("used") - order["amount"])
            self.balance["USDT"]["free"] += order["amount"] * price
        else:
            self.contracts = round(self.contracts + (-order["amount"] if order["side"] == "sell" else order["amount"]), 10)

    async def fetch_order(self, order_id, symbol):
        if order_id not in self.orders:
            raise ccxt.OrderNotFound(order_id)
        return dict(self.orders[order_id])

    async def fetch_open_orders(self, symbol=None):
        return [dict(o) for o in self.orders.values() if o["status"] == "open"]

    async def fetch_closed_orders(self, symbol=None):
        return [dict(o) for o in self.orders.values() if o["status"] != "open"]

    async def fetch_balance(self):
        return {k: dict(v) for k, v in self.balance.items()}

    async def fetch_positions(self, symbols=None):
        return [{"symbol": "BTC/USDT", "contracts": abs(self.contracts)}]

    async def set_leverage(self, leverage, symbol):
        self.calls.append(("leverage", leverage, symbol))

    async def set_margin_mode(self, mode, symbol):
        self.calls.append(("margin", mode, symbol))

    async def close(self):
        return None

    def entries(self):
        return [o for o in self.created if not o["trigger"] and o["side"] == "buy"]


def _live_settings(tmp_path, **overrides):
    defaults = dict(
        paper_trading=False,
        live_trading_confirm=True,
        exchange_api_key="key",
        exchange_api_secret="secret",
        market_type="spot",
        fee_rate=0.0,
        retry_attempts=3,
        retry_base_delay=0.001,
        order_poll_attempts=2,
        order_poll_delay=0.001,
        data_dir=str(tmp_path),
        cooldown_minutes=30,
    )
    defaults.update(overrides)
    return make_settings(**defaults)


def _store(tmp_path, settings, clock=None):
    return PortfolioStore(
        tmp_path / "live.sqlite",
        10_000,
        fee_rate=0.0,
        clock=clock or (lambda: NOW),
        market_type=settings.market_type,
        leverage=settings.leverage,
    )


def _decision(entry=100.0) -> RiskDecision:
    return RiskDecision(
        action=RiskAction.EXECUTE_BUY,
        symbol="BTC/USDT",
        position_size=0.2,
        entry=entry,
        stop_loss=entry - 3,
        take_profit=entry + 6,
        confidence=0.8,
        reasoning="live test",
    )


def _sell_signal_decision() -> RiskDecision:
    return RiskDecision(
        action=RiskAction.EXECUTE_SELL,
        symbol="BTC/USDT",
        position_size=0.2,
        entry=104,
        stop_loss=0,
        take_profit=0,
        confidence=0.7,
        reasoning="exit",
    )


@pytest.mark.asyncio
async def test_public_client_enables_rate_limit() -> None:
    exchange = build_ccxt_exchange(make_settings(), authenticated=False)
    assert exchange.enableRateLimit is True
    await exchange.close()


def test_live_exchange_refuses_paper_mode(tmp_path) -> None:
    settings = make_settings(paper_trading=True)
    store = PortfolioStore(tmp_path / "nope.sqlite", 1000)
    with pytest.raises(LiveTradingNotConfirmedError):
        LiveExchange(settings, FakeExchange(), store)
    store.close()


@pytest.mark.asyncio
async def test_lost_response_does_not_double_enter(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.lose_response = 1  # the entry reaches the book, the reply times out
    live = LiveExchange(settings, fake, store)
    result = await live.execute(_decision())
    assert result.success and result.opened
    assert len(fake.entries()) == 1, "the order must be found by clientOrderId, not resent"
    assert fake.entries()[0]["clientOrderId"].startswith("ctb")
    await live.close()


@pytest.mark.asyncio
async def test_network_errors_before_the_book_are_retried_once_each(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.fail_before = [ccxt.NetworkError("down"), ccxt.NetworkError("down")]
    live = LiveExchange(settings, fake, store)
    result = await live.execute(_decision())
    assert result.success and result.opened
    assert len(fake.entries()) == 1
    await live.close()


@pytest.mark.asyncio
async def test_insufficient_funds_is_not_retried(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.fail_before = [ccxt.InsufficientFunds("nope")] * 3
    live = LiveExchange(settings, fake, store)
    result = await live.execute(_decision())
    assert result.success is False
    assert result.status == "insufficient_funds"
    assert len(fake.fail_before) == 2  # exactly one attempt consumed
    assert store.get_position("BTC/USDT") is None
    await live.close()


@pytest.mark.asyncio
async def test_entry_places_a_tracked_stop_from_the_real_fill(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.market_price = 101.0  # one point of slippage against the planned 100
    live = LiveExchange(settings, fake, store)
    result = await live.execute(_decision(entry=100.0))
    assert result.success and result.opened
    position = store.get_position("BTC/USDT")
    assert position.entry == pytest.approx(101)
    assert position.stop_loss == pytest.approx(98)  # distance 3 kept from the fill
    assert position.take_profit == pytest.approx(107)
    assert position.exchange_protected is True
    stop = fake.orders[position.sl_order_id]
    assert stop["side"] == "sell" and stop["status"] == "open"
    assert stop["params"]["stopLossPrice"] == pytest.approx(98)
    # Spot: no exchange take-profit, which would lock the same coins twice.
    assert position.tp_order_id is None
    await live.close()


@pytest.mark.asyncio
async def test_stop_type_fallback_when_unified_param_is_refused(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.rejected_types = {"stopLossPrice"}
    live = LiveExchange(settings, fake, store)
    await live.execute(_decision())
    position = store.get_position("BTC/USDT")
    assert position.exchange_protected is True
    assert fake.orders[position.sl_order_id]["trigger"] == "triggerPrice"
    await live.close()


@pytest.mark.asyncio
async def test_signal_close_cancels_the_stop_and_sells_what_is_held(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.base_fee = 0.001  # 0.2 bought, 0.1998 received
    live = LiveExchange(settings, fake, store)
    await live.execute(_decision())
    stop_id = store.get_position("BTC/USDT").sl_order_id
    fake.market_price = 104.0
    result = await live.execute(_sell_signal_decision(), market_price=104)
    assert result.success, result.message
    assert stop_id in fake.cancelled
    sells = [o for o in fake.created if o["side"] == "sell" and not o["trigger"]]
    assert sells and sells[-1]["amount"] == pytest.approx(0.1998)
    assert result.closed_trade.reason == CloseReason.SIGNAL
    assert store.get_position("BTC/USDT") is None
    assert not [o for o in fake.orders.values() if o["status"] == "open"], "no orphan orders"
    await live.close()


@pytest.mark.asyncio
async def test_exchange_stop_fill_uses_the_stop_order_price(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    live = LiveExchange(settings, fake, store)
    await live.execute(_decision())
    position = store.get_position("BTC/USDT")
    fake.trigger(position.sl_order_id, 96.5)  # slipped below the 97 trigger
    marked = await live.mark_prices({"BTC/USDT": 99.0})  # price already bounced
    assert len(marked.closed) == 1
    trade = marked.closed[0]
    assert trade.reason == CloseReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(96.5)
    assert "BTC/USDT" in store.view().cooldowns
    await live.close()


@pytest.mark.asyncio
async def test_spot_take_profit_is_managed_by_the_bot(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    live = LiveExchange(settings, fake, store)
    await live.execute(_decision())
    stop_id = store.get_position("BTC/USDT").sl_order_id
    fake.market_price = 106.5
    marked = await live.mark_prices({"BTC/USDT": 106.5})
    assert len(marked.closed) == 1
    assert marked.closed[0].reason == CloseReason.TAKE_PROFIT
    assert stop_id in fake.cancelled
    await live.close()


@pytest.mark.asyncio
async def test_software_stop_when_no_stop_type_is_accepted(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.rejected_types = {"stopLossPrice", "triggerPrice"}
    live = LiveExchange(settings, fake, store)
    result = await live.execute(_decision())
    assert "WARNING" in result.message
    assert store.get_position("BTC/USDT").exchange_protected is False
    fake.market_price = 90.0
    marked = await live.mark_prices({"BTC/USDT": 90.0})
    assert len(marked.closed) == 1
    assert marked.closed[0].reason == CloseReason.STOP_LOSS
    await live.close()


@pytest.mark.asyncio
async def test_backstop_closes_when_price_stays_beyond_a_protected_stop(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    live = LiveExchange(settings, fake, store)
    await live.execute(_decision())
    fake.market_price = 95.0  # the exchange stop "did not fire"
    first = await live.mark_prices({"BTC/USDT": 95.0})
    assert not first.closed
    assert any("beyond the stop" in e for e in first.errors)
    second = await live.mark_prices({"BTC/USDT": 95.0})
    assert len(second.closed) == 1
    assert second.closed[0].reason == CloseReason.STOP_LOSS
    await live.close()


@pytest.mark.asyncio
async def test_reconcile_replaces_missing_stop_and_cancels_orphans(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    live = LiveExchange(settings, fake, store)
    await live.execute(_decision())
    position = store.get_position("BTC/USDT")
    # Someone cancelled our stop by hand, and an orphan bot order is resting.
    await fake.cancel_order(position.sl_order_id, "BTC/USDT")
    fake.orders["99"] = {
        "id": "99", "clientOrderId": "ctbdeadbeef", "symbol": "BTC/USDT", "type": "limit", "side": "buy",
        "amount": 0.1, "price": 80, "params": {}, "trigger": None, "status": "open", "filled": 0.0, "average": None,
    }
    notes = await live.reconcile(["BTC/USDT"])
    replaced = store.get_position("BTC/USDT")
    assert replaced.sl_order_id and replaced.sl_order_id != position.sl_order_id
    assert fake.orders[replaced.sl_order_id]["status"] == "open"
    assert "99" in fake.cancelled
    assert any("orphan" in note for note in notes)
    await live.close()


@pytest.mark.asyncio
async def test_external_close_is_detected(tmp_path) -> None:
    settings = _live_settings(tmp_path)
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    live = LiveExchange(settings, fake, store)
    await live.execute(_decision())
    position = store.get_position("BTC/USDT")
    await fake.cancel_order(position.sl_order_id, "BTC/USDT")
    fake._set_btc(free=0.0, used=0.0)  # sold manually on the exchange website
    marked = await live.mark_prices({"BTC/USDT": 101.0})
    assert len(marked.closed) == 1
    assert marked.closed[0].reason == CloseReason.EXTERNAL
    assert any("no bot order explains it" in e for e in marked.errors)
    await live.close()


@pytest.mark.asyncio
async def test_resting_limit_expires_and_is_cancelled(tmp_path) -> None:
    clock = {"now": NOW}
    settings = _live_settings(tmp_path, order_type="limit", pending_order_ttl_minutes=30)
    store = _store(tmp_path, settings, clock=lambda: clock["now"])
    fake = FakeExchange()
    fake.fill_limits = False
    live = LiveExchange(settings, fake, store)
    result = await live.execute(_decision())
    assert result.status == "open"
    assert len(store.pending_orders()) == 1
    clock["now"] = NOW + timedelta(minutes=31)
    marked = await live.mark_prices({"BTC/USDT": 101})
    assert store.pending_orders() == []
    assert any("expired" in e for e in marked.errors)
    assert result.order_id in fake.cancelled
    await live.close()


@pytest.mark.asyncio
async def test_resting_limit_fill_gets_protection(tmp_path) -> None:
    settings = _live_settings(tmp_path, order_type="limit")
    store = _store(tmp_path, settings)
    fake = FakeExchange()
    fake.fill_limits = False
    live = LiveExchange(settings, fake, store)
    result = await live.execute(_decision())
    order = fake.orders[result.order_id]
    order.update(status="closed", filled=order["amount"], average=100.0)
    fake._set_btc(free=0.2)
    marked = await live.mark_prices({"BTC/USDT": 100.5})
    assert len(marked.opened) == 1
    position = store.get_position("BTC/USDT")
    assert position.exchange_protected and position.sl_order_id
    await live.close()


@pytest.mark.asyncio
async def test_swap_sets_leverage_and_places_reduce_only_target(tmp_path) -> None:
    settings = _live_settings(tmp_path, market_type="swap", leverage=2)
    store = _store(tmp_path, settings)
    fake = FakeExchange(market_type="swap")
    live = LiveExchange(settings, fake, store)
    await live.reconcile(["BTC/USDT"])
    assert ("leverage", 2, "BTC/USDT") in fake.calls
    assert ("margin", "isolated", "BTC/USDT") in fake.calls
    await live.execute(_decision())
    position = store.get_position("BTC/USDT")
    assert position.margin_locked == pytest.approx(0.2 * 100 / 2)
    tp = fake.orders[position.tp_order_id]
    assert tp["params"]["reduceOnly"] is True
    fake.trigger(position.tp_order_id, 106)
    marked = await live.mark_prices({"BTC/USDT": 106})
    assert marked.closed[0].reason == CloseReason.TAKE_PROFIT
    assert position.sl_order_id in fake.cancelled
    await live.close()
