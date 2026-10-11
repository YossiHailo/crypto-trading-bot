"""Local web dashboard. Standard library only (asyncio streams), no extra dependency.

Open http://127.0.0.1:8765 while the bot runs. The page shows the account,
open positions, the latest signals with their reasoning, every risk decision,
closed trades, warnings, and an equity curve. Buttons let the operator pause
new entries, resume, run a cycle now, reset the kill switch, close every
position, or stop the bot.

Security model:

* Binds to 127.0.0.1 by default. Settings refuse a non-loopback bind without
  ``DASHBOARD_TOKEN``.
* Every POST must carry ``X-Requested-With: ctb-dashboard`` (a custom header a
  foreign web page cannot send without a CORS preflight, which this server never
  approves) and, when configured, ``X-Dashboard-Token``.
* On a loopback bind the ``Host`` header must be a loopback name, which blocks
  DNS-rebinding pages from reading ``/api/state``.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import Settings, _is_loopback
from core.control import Controller
from core.redact import redact
from core.state import next_deposit_at

logger = logging.getLogger("trading.dashboard")

_STATIC = Path(__file__).resolve().parent / "static"
_MAX_BODY = 16_384
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
_ACTIONS = {"pause", "resume", "run-now", "reset-halt", "close-all", "stop"}


def json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _pct(numerator: float, denominator: float) -> float | None:
    return (numerator / denominator) if denominator else None


@dataclass
class DashboardContext:
    settings: Settings
    store: Any
    execution: Any
    controller: Controller
    notifier: Any = None


def build_state(ctx: DashboardContext) -> dict[str, Any]:
    """Everything the page renders, in one JSON document."""
    settings, store, status = ctx.settings, ctx.store, ctx.controller.status
    marks = status.last_prices
    view = store.view(marks)
    positions = []
    for position in view.positions.values():
        mark = marks.get(position.symbol, position.entry)
        unrealized = position.unrealized(mark)
        span = position.take_profit - position.stop_loss
        progress = (mark - position.stop_loss) / span if span else 0.5
        if position.side == "short":
            progress = 1 - progress
        positions.append(
            {
                "symbol": position.symbol,
                "side": position.side,
                "size": position.size,
                "entry": position.entry,
                "mark": mark,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "unrealized": unrealized,
                "unrealized_pct": _pct(unrealized, position.size * position.entry),
                "to_stop_pct": _pct(abs(mark - position.stop_loss), mark),
                "to_target_pct": _pct(abs(position.take_profit - mark), mark),
                "opened_at": position.opened_at,
                "exchange_protected": position.exchange_protected,
                # Money lost / won from here if the stop / target is hit (before fees).
                "loss_at_stop": abs(mark - position.stop_loss) * position.size,
                "gain_at_target": abs(position.take_profit - mark) * position.size,
                "progress": max(0.0, min(1.0, progress)),
                "notional": position.size * mark,
            }
        )
    day_dd = _pct(view.day_start_equity - view.equity, view.day_start_equity) or 0.0
    # Money the investor put in (starting balance + simulated monthly transfers).
    # Profit is measured against this, never against equity that includes deposits.
    deposited = store.total_deposits() if hasattr(store, "total_deposits") else 0.0
    baseline = deposited or store.first_equity() or settings.starting_balance
    monthly = settings.monthly_deposit if settings.paper_trading else 0.0
    peak = max(view.peak_equity, view.equity)
    peak_dd = _pct(peak - view.equity, peak) or 0.0
    return {
        "config": settings.public_summary(),
        "status": {
            "mode": status.mode,
            "started_at": status.started_at,
            "cycle_count": status.cycle_count,
            "cycle_running": status.cycle_running,
            "last_cycle_started": status.last_cycle_started,
            "last_cycle_finished": status.last_cycle_finished,
            "next_cycle_at": status.next_cycle_at,
            "stopping": status.stopping,
            "paused": view.paused,
            "halted_reason": view.halted_reason,
            "balance_synced": view.balance_synced,
        },
        "account": {
            "cash": view.cash,
            "equity": view.equity,
            "day_start_equity": view.day_start_equity,
            "day_pnl": view.equity - view.day_start_equity,
            "day_drawdown_pct": max(0.0, day_dd),
            "daily_limit_pct": settings.daily_max_loss_pct,
            "peak_equity": peak,
            "peak_drawdown_pct": max(0.0, peak_dd),
            "max_drawdown_pct": settings.max_drawdown_pct,
            "open_notional": view.open_notional,
            "exposure_pct": _pct(view.open_notional, view.equity) or 0.0,
            "quote": view.quote_currency,
            "baseline": baseline,
            "total_change": view.equity - baseline,
            "total_change_pct": _pct(view.equity - baseline, baseline),
            "total_deposits": baseline,
            "monthly_deposit": monthly,
            "next_deposit_at": next_deposit_at(store.now(), settings.deposit_day) if monthly > 0 else None,
        },
        "deposits": store.deposits() if hasattr(store, "deposits") else [],
        "activity_24h": store.decision_counts(24),
        "stats": store.trade_stats(),
        "positions": positions,
        "pending": [
            {
                "symbol": order.symbol,
                "side": order.side,
                "size": order.size,
                "limit_price": order.limit_price,
                "created_at": order.created_at,
            }
            for order in store.pending_orders()
        ],
        "cooldowns": view.cooldowns,
        "signals": status.last_signals,
        "sentiment": status.last_sentiment,
        "errors": status.last_errors,
        "decisions": store.recent_decisions(40),
        "trades": store.recent_trades(50),
        "events": store.recent_events(40),
        "equity": store.equity_history(1500),
    }


async def perform_action(ctx: DashboardContext, action: str) -> dict[str, Any]:
    store, controller = ctx.store, ctx.controller
    marks = controller.status.last_prices
    if action == "pause":
        store.set_paused(True)
        message = "New entries paused from the dashboard. Stops and exits keep running."
    elif action == "resume":
        store.set_paused(False)
        message = "New entries resumed from the dashboard."
    elif action == "reset-halt":
        store.reset_halt(marks)
        message = "Kill switch reset from the dashboard. The equity peak starts again from now."
    elif action == "run-now":
        controller.request_cycle()
        message = "A cycle was requested."
    elif action == "stop":
        controller.request_stop()
        message = "Stop requested. The bot finishes the current step and exits."
    elif action == "close-all":
        async with controller.cycle_lock:
            results = await ctx.execution.close_all(marks)
        closed = sum(1 for result in results if result.closed_trade is not None)
        failed = [redact(result.message) for result in results if not result.success]
        message = f"Close-all from the dashboard: {closed} position(s) closed."
        if failed:
            message += " Failures: " + "; ".join(failed)
    else:
        raise ValueError(action)
    store.add_event("warning" if action in {"close-all", "stop"} else "info", message)
    logger.warning(message)
    if ctx.notifier is not None:
        try:
            await ctx.notifier.info(message)
        except Exception:
            logger.debug("notifier failed", exc_info=True)
    return {"ok": True, "message": message}


class DashboardServer:
    def __init__(self, ctx: DashboardContext) -> None:
        self.ctx = ctx
        self._server: asyncio.base_events.Server | None = None
        self.port: int | None = None

    async def start(self) -> None:
        settings = self.ctx.settings
        self._server = await asyncio.start_server(self._handle, settings.dashboard_host, settings.dashboard_port)
        sockets = self._server.sockets or []
        self.port = sockets[0].getsockname()[1] if sockets else settings.dashboard_port
        logger.info("Dashboard on http://%s:%s", settings.dashboard_host, self.port)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Only reading the request is time-limited. An action such as close-all
            # must never be cancelled halfway because a client was slow.
            request = await asyncio.wait_for(self._read_request(reader), timeout=15)
            if isinstance(request, tuple) and len(request) == 3 and isinstance(request[0], int):
                status, headers, body = request
            else:
                status, headers, body = await self._dispatch(*request)
        except asyncio.TimeoutError:
            status, headers, body = 408, {"Content-Type": "text/plain"}, b"timeout"
        except Exception:
            logger.exception("Dashboard request failed")
            status, headers, body = 500, {"Content-Type": "text/plain"}, b"internal error"
        reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                  408: "Request Timeout", 413: "Payload Too Large", 500: "Internal Server Error"}.get(status, "OK")
        head = [f"HTTP/1.1 {status} {reason}", f"Content-Length: {len(body)}", "Connection: close",
                "Cache-Control: no-store", "X-Content-Type-Options: nosniff", "Referrer-Policy: no-referrer",
                "X-Frame-Options: DENY"]
        head += [f"{key}: {value}" for key, value in headers.items()]
        try:
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body)
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()

    async def _read_request(self, reader: asyncio.StreamReader) -> tuple:
        """Return ``(method, path, headers)`` or an error response tuple."""
        request_line = (await reader.readline()).decode("latin-1").strip()
        parts = request_line.split()
        if len(parts) != 3:
            return 400, {"Content-Type": "text/plain"}, b"bad request"
        method, target, _version = parts
        headers: dict[str, str] = {}
        while True:
            line = (await reader.readline()).decode("latin-1")
            if line in {"\r\n", "\n", ""}:
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
            if len(headers) > 100:
                return 400, {"Content-Type": "text/plain"}, b"too many headers"
        length = int(headers.get("content-length", "0") or 0)
        if length > _MAX_BODY:
            return 413, {"Content-Type": "text/plain"}, b"body too large"
        if length:
            await reader.readexactly(length)
        return method, target.split("?", 1)[0], headers

    async def _dispatch(self, method: str, path: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        if not self._host_allowed(headers.get("host", "")):
            return 403, {"Content-Type": "text/plain"}, b"host not allowed"

        if method == "GET" and path in {"/", "/index.html"}:
            return 200, {"Content-Type": "text/html; charset=utf-8"}, (_STATIC / "index.html").read_bytes()
        if method == "GET" and path == "/api/state":
            if not self._token_ok(headers):
                return 403, {"Content-Type": "application/json"}, b'{"error":"token required"}'
            payload = json.dumps(build_state(self.ctx), default=json_default)
            return 200, {"Content-Type": "application/json"}, redact(payload).encode("utf-8")
        if path.startswith("/api/control/"):
            if method != "POST":
                return 405, {"Content-Type": "text/plain", "Allow": "POST"}, b"POST only"
            if headers.get("x-requested-with") != "ctb-dashboard" or not self._token_ok(headers):
                return 403, {"Content-Type": "application/json"}, b'{"error":"forbidden"}'
            action = path.rsplit("/", 1)[-1]
            if action not in _ACTIONS:
                return 404, {"Content-Type": "application/json"}, b'{"error":"unknown action"}'
            result = await perform_action(self.ctx, action)
            return 200, {"Content-Type": "application/json"}, json.dumps(result).encode("utf-8")
        return 404, {"Content-Type": "text/plain"}, b"not found"

    def _host_allowed(self, host_header: str) -> bool:
        if not _is_loopback(self.ctx.settings.dashboard_host):
            return True  # token-protected remote bind (for example inside Docker)
        host = host_header.rsplit(":", 1)[0] if not host_header.startswith("[") else host_header.split("]")[0] + "]"
        return host in _LOOPBACK_HOSTS

    def _token_ok(self, headers: dict[str, str]) -> bool:
        token = self.ctx.settings.dashboard_token
        if not token:
            return True
        return hmac.compare_digest(headers.get("x-dashboard-token", ""), token)
