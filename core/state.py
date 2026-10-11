"""SQLite state shared by the paper simulator, the live executor and the dashboard.

What lives here:

* the account (cash, the UTC-day baseline for the daily loss breaker, the
  equity peak for the max-drawdown kill switch, pause/halt flags),
* positions the bot opened, with the exchange ids of their protective orders,
* resting entry orders, cooldowns and closed trades,
* every risk decision and an equity snapshot per cycle, so behaviour can be
  audited and charted later.

In live mode the exchange is the source of truth. Cash is overwritten with the
exchange balance each cycle and ``LiveExchange.reconcile`` compares positions
and open orders with the exchange at startup.

Accounting has two models. A spot long holds the coin, so cash is debited by
the full notional. A short, or any swap position, locks margin
(``notional / leverage``) and equity is ``margin + unrealized PnL``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.errors import InsufficientFundsError
from core.models import (
    CloseReason,
    ClosedTrade,
    PendingOrder,
    PortfolioView,
    Position,
    RiskDecision,
    TechnicalSignal,
    utcnow,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL,
    day_start_equity REAL NOT NULL,
    day_start_date TEXT NOT NULL,
    realized_pnl_today REAL NOT NULL,
    balance_synced INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    side TEXT NOT NULL,
    size REAL NOT NULL,
    entry REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    opened_at TEXT NOT NULL,
    open_fee REAL NOT NULL,
    margin_locked REAL NOT NULL,
    exchange_protected INTEGER NOT NULL,
    order_id TEXT
);
CREATE TABLE IF NOT EXISTS pending_orders (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size REAL NOT NULL,
    limit_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cooldowns (
    symbol TEXT PRIMARY KEY,
    until_ts TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size REAL NOT NULL,
    entry REAL NOT NULL,
    exit_price REAL NOT NULL,
    pnl REAL NOT NULL,
    reason TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    order_id TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    signal TEXT NOT NULL,
    signal_confidence REAL NOT NULL,
    sentiment_score REAL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    reasoning TEXT NOT NULL,
    indicators TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions (ts);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    open_positions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS deposits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    amount REAL NOT NULL,
    note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
"""

# Columns added after the first release. ``_migrate`` adds any that are missing
# so an existing ``portfolio.sqlite`` keeps working after an upgrade.
_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "account": [
        ("started_at", "TEXT"),
        ("total_deposits", "REAL"),
        ("peak_equity", "REAL NOT NULL DEFAULT 0"),
        ("halted_reason", "TEXT"),
        ("paused", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "positions": [
        ("sl_order_id", "TEXT"),
        ("tp_order_id", "TEXT"),
    ],
    "pending_orders": [
        ("client_order_id", "TEXT"),
    ],
}


def _dt_to_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _text_to_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def date_first_of_next_month(year: int, month: int) -> datetime:
    return datetime(year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=timezone.utc)


def next_deposit_at(now: datetime, day_of_month: int) -> datetime:
    """The next scheduled monthly deposit strictly after ``now`` (00:00 UTC, day clamped)."""
    year, month = now.year, now.month
    while True:
        last_day = (date_first_of_next_month(year, month) - timedelta(days=1)).day
        when = datetime(year, month, min(day_of_month, last_day), tzinfo=timezone.utc)
        if when > now:
            return when
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def _uses_margin(position: Position) -> bool:
    return position.side == "short" or position.margin_locked > 0


def _equity(cash: float, positions: list[Position], marks: dict[str, float] | None) -> float:
    equity = cash
    for position in positions:
        mark = position.entry
        if marks and position.symbol in marks:
            mark = marks[position.symbol]
        if _uses_margin(position):
            equity += position.margin_locked + position.unrealized(mark)
        else:
            # Spot long: cash was reduced by the entry notional, so add market value back.
            equity += position.size * mark
    return equity


class PortfolioStore:
    """Persistent account used by risk checks, both execution backends and the dashboard."""

    def __init__(
        self,
        path: str | Path,
        starting_balance: float,
        *,
        quote_currency: str = "USDT",
        fee_rate: float = 0.001,
        exchange_balance_tracking: bool = False,
        clock: Callable[[], datetime] | None = None,
        market_type: str = "spot",
        leverage: float = 1.0,
        max_drawdown_pct: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.quote_currency = quote_currency
        self.fee_rate = fee_rate
        self.market_type = market_type
        self.leverage = leverage if market_type == "swap" else 1.0
        self.max_drawdown_pct = max_drawdown_pct
        self._clock = clock or utcnow
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._ensure_account(starting_balance, exchange_balance_tracking)
        self._conn.commit()

    # ------------------------------------------------------------------ setup

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> datetime:
        return self._clock()

    def now(self) -> datetime:
        return self._clock()

    def _migrate(self) -> None:
        for table, columns in _MIGRATIONS.items():
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, ddl in columns:
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def _ensure_account(self, starting_balance: float, exchange_balance_tracking: bool) -> None:
        row = self._conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
        if row is not None:
            if row["started_at"] is None or row["total_deposits"] is None:
                # Databases created before deposits existed: the start is the first snapshot.
                first = self._conn.execute("SELECT MIN(ts) AS ts FROM equity_snapshots").fetchone()
                started = first["ts"] if first and first["ts"] else _dt_to_text(self._now())
                self._conn.execute(
                    "UPDATE account SET started_at = COALESCE(started_at, ?), total_deposits = COALESCE(total_deposits, ?) WHERE id = 1",
                    (started, starting_balance),
                )
            return
        today = self._now().date().isoformat()
        # Live mode stays unsynced until the exchange balance arrives, so the
        # risk agent will not size off the configured paper starting balance.
        synced = 0 if exchange_balance_tracking else 1
        self._conn.execute(
            """
            INSERT INTO account (
                id, cash, day_start_equity, day_start_date, realized_pnl_today,
                balance_synced, peak_equity, halted_reason, paused, started_at, total_deposits
            ) VALUES (1, ?, ?, ?, 0, ?, ?, NULL, 0, ?, ?)
            """,
            (starting_balance, starting_balance, today, synced, starting_balance, _dt_to_text(self._now()), starting_balance),
        )
        self._conn.execute(
            "INSERT INTO deposits (ts, amount, note) VALUES (?, ?, 'initial')",
            (_dt_to_text(self._now()), starting_balance),
        )

    def _account(self) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError("account row is missing")
        return row

    # -------------------------------------------------------------- positions

    def _load_positions(self) -> list[Position]:
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        return [self._position_from_row(row) for row in rows]

    def _position_from_row(self, row: sqlite3.Row) -> Position:
        return Position(
            symbol=row["symbol"],
            side=row["side"],
            size=float(row["size"]),
            entry=float(row["entry"]),
            stop_loss=float(row["stop_loss"]),
            take_profit=float(row["take_profit"]),
            opened_at=_text_to_dt(row["opened_at"]),
            open_fee=float(row["open_fee"]),
            margin_locked=float(row["margin_locked"]),
            exchange_protected=bool(row["exchange_protected"]),
            order_id=row["order_id"],
            sl_order_id=row["sl_order_id"],
            tp_order_id=row["tp_order_id"],
        )

    def get_position(self, symbol: str) -> Position | None:
        row = self._conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()
        if row is None:
            return None
        return self._position_from_row(row)

    def positions(self) -> list[Position]:
        return self._load_positions()

    def pending_orders(self) -> list[PendingOrder]:
        rows = self._conn.execute("SELECT * FROM pending_orders").fetchall()
        return [
            PendingOrder(
                id=row["id"],
                symbol=row["symbol"],
                side=row["side"],
                size=float(row["size"]),
                limit_price=float(row["limit_price"]),
                stop_loss=float(row["stop_loss"]),
                take_profit=float(row["take_profit"]),
                created_at=_text_to_dt(row["created_at"]),
                client_order_id=row["client_order_id"],
            )
            for row in rows
        ]

    def view(self, marks: dict[str, float] | None = None) -> PortfolioView:
        account = self._account()
        positions = {position.symbol: position for position in self._load_positions()}
        now = self._now()
        cooldowns: dict[str, datetime] = {}
        for row in self._conn.execute("SELECT symbol, until_ts FROM cooldowns").fetchall():
            until = _text_to_dt(row["until_ts"])
            if until > now:
                cooldowns[row["symbol"]] = until
        pending = {order.symbol for order in self.pending_orders()}
        cash = float(account["cash"])
        open_notional = 0.0
        for position in positions.values():
            mark = (marks or {}).get(position.symbol, position.entry)
            open_notional += position.size * mark
        for order in self.pending_orders():
            open_notional += order.size * order.limit_price
        return PortfolioView(
            cash=cash,
            equity=_equity(cash, list(positions.values()), marks),
            day_start_equity=float(account["day_start_equity"]),
            realized_pnl_today=float(account["realized_pnl_today"]),
            day_start_date=str(account["day_start_date"]),
            positions=positions,
            cooldowns=cooldowns,
            pending_symbols=pending,
            quote_currency=self.quote_currency,
            balance_synced=bool(account["balance_synced"]),
            peak_equity=float(account["peak_equity"] or 0.0),
            halted_reason=account["halted_reason"],
            paused=bool(account["paused"]),
            open_notional=open_notional,
        )

    def sync_quote_balance(self, free_quote: float, marks: dict[str, float] | None = None) -> None:
        """Replace local cash with the exchange free-quote balance.

        The first successful sync also rebases the circuit-breaker day equity
        and the drawdown peak so a live account is not judged against
        ``STARTING_BALANCE``.
        """
        positions = self._load_positions()
        equity = _equity(free_quote, positions, marks)
        account = self._account()
        if int(account["balance_synced"]) == 0:
            self._conn.execute(
                """
                UPDATE account
                SET cash = ?, day_start_equity = ?, peak_equity = ?, balance_synced = 1
                WHERE id = 1
                """,
                (free_quote, equity, equity),
            )
        else:
            self._conn.execute("UPDATE account SET cash = ? WHERE id = 1", (free_quote,))
        self._conn.commit()

    def add_pending(
        self,
        *,
        symbol: str,
        side: str,
        size: float,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
        created_at: datetime | None = None,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> PendingOrder:
        created_at = created_at or self._now()
        pending = PendingOrder(
            id=order_id or uuid.uuid4().hex,
            symbol=symbol,
            side=side,
            size=size,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            created_at=created_at,
            client_order_id=client_order_id,
        )
        self._conn.execute(
            """
            INSERT INTO pending_orders (
                id, symbol, side, size, limit_price, stop_loss, take_profit, created_at, client_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pending.id,
                pending.symbol,
                pending.side,
                pending.size,
                pending.limit_price,
                pending.stop_loss,
                pending.take_profit,
                _dt_to_text(pending.created_at),
                pending.client_order_id,
            ),
        )
        self._conn.commit()
        return pending

    def remove_pending(self, order_id: str) -> None:
        self._conn.execute("DELETE FROM pending_orders WHERE id = ?", (order_id,))
        self._conn.commit()

    def open_position(
        self,
        *,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        order_id: str | None = None,
        exchange_protected: bool = False,
        opened_at: datetime | None = None,
        sl_order_id: str | None = None,
        tp_order_id: str | None = None,
        enforce_cash: bool = True,
    ) -> Position:
        """Debit cash and store the position. Raises ``InsufficientFundsError``.

        ``enforce_cash=False`` is for fills that already happened on the
        exchange: the position must be recorded even if local cash disagrees.
        """
        if size <= 0 or price <= 0:
            raise ValueError("size and price must be positive")
        if side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        notional = size * price
        open_fee = notional * self.fee_rate
        if self.market_type == "swap":
            margin_locked = notional / self.leverage
        elif side == "short":
            margin_locked = notional
        else:
            margin_locked = 0.0
        debit = (margin_locked + open_fee) if margin_locked > 0 else (notional + open_fee)
        account = self._account()
        cash = float(account["cash"])
        if enforce_cash and cash + 1e-9 < debit:
            raise InsufficientFundsError(f"need {debit:.4f} {self.quote_currency}, have {cash:.4f}")
        opened_at = opened_at or self._now()
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=opened_at,
            open_fee=open_fee,
            margin_locked=margin_locked,
            exchange_protected=exchange_protected,
            order_id=order_id,
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
        )
        self._conn.execute("UPDATE account SET cash = ? WHERE id = 1", (cash - debit,))
        self._conn.execute(
            """
            INSERT OR REPLACE INTO positions (
                symbol, side, size, entry, stop_loss, take_profit, opened_at,
                open_fee, margin_locked, exchange_protected, order_id, sl_order_id, tp_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.symbol,
                position.side,
                position.size,
                position.entry,
                position.stop_loss,
                position.take_profit,
                _dt_to_text(position.opened_at),
                position.open_fee,
                position.margin_locked,
                int(position.exchange_protected),
                position.order_id,
                position.sl_order_id,
                position.tp_order_id,
            ),
        )
        self._conn.commit()
        return position

    def update_protection(
        self,
        symbol: str,
        *,
        exchange_protected: bool,
        sl_order_id: str | None,
        tp_order_id: str | None,
    ) -> None:
        self._conn.execute(
            "UPDATE positions SET exchange_protected = ?, sl_order_id = ?, tp_order_id = ? WHERE symbol = ?",
            (int(exchange_protected), sl_order_id, tp_order_id, symbol),
        )
        self._conn.commit()

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: CloseReason,
        *,
        closed_at: datetime | None = None,
    ) -> ClosedTrade:
        position = self.get_position(symbol)
        if position is None:
            raise KeyError(symbol)
        if exit_price <= 0:
            raise ValueError("exit_price must be positive")
        closed_at = closed_at or self._now()
        close_fee = position.size * exit_price * self.fee_rate
        account = self._account()
        cash = float(account["cash"])
        if _uses_margin(position):
            gross = position.unrealized(exit_price)
            cash += position.margin_locked + gross - close_fee
            pnl = gross - position.open_fee - close_fee
        else:
            proceeds = position.size * exit_price - close_fee
            cash += proceeds
            pnl = proceeds - (position.size * position.entry) - position.open_fee
        realized = float(account["realized_pnl_today"]) + pnl
        self._conn.execute(
            "UPDATE account SET cash = ?, realized_pnl_today = ? WHERE id = 1",
            (cash, realized),
        )
        self._conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        self._conn.execute(
            """
            INSERT INTO trades (
                symbol, side, size, entry, exit_price, pnl, reason, opened_at, closed_at, order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.symbol,
                position.side,
                position.size,
                position.entry,
                exit_price,
                pnl,
                reason.value,
                _dt_to_text(position.opened_at),
                _dt_to_text(closed_at),
                position.order_id,
            ),
        )
        self._conn.commit()
        return ClosedTrade(
            symbol=position.symbol,
            side=position.side,
            size=position.size,
            entry=position.entry,
            exit_price=exit_price,
            pnl=pnl,
            reason=reason,
            opened_at=position.opened_at,
            closed_at=closed_at,
            order_id=position.order_id,
        )

    def set_cooldown(self, symbol: str, until: datetime, reason: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cooldowns (symbol, until_ts, reason) VALUES (?, ?, ?)",
            (symbol, _dt_to_text(until), reason),
        )
        self._conn.commit()

    # ------------------------------------------------- day roll, peak, halts

    def roll_day_if_needed(self, marks: dict[str, float] | None = None) -> str | None:
        """Start a new UTC day. Returns a summary of the day that just ended."""
        account = self._account()
        today = self._now().date().isoformat()
        previous = str(account["day_start_date"])
        if previous == today:
            return None
        summary = self._format_day_summary(previous, marks)
        equity = self.view(marks).equity
        self._conn.execute(
            """
            UPDATE account
            SET day_start_date = ?, day_start_equity = ?, realized_pnl_today = 0
            WHERE id = 1
            """,
            (today, equity),
        )
        self._conn.commit()
        return summary

    def update_peak_and_check_drawdown(self, marks: dict[str, float] | None = None) -> str | None:
        """Track the equity high-water mark. Returns a halt reason when it trips.

        Unlike the daily breaker, this halt does not reset by itself. Someone
        has to look at what happened and reset it from the dashboard.
        """
        view = self.view(marks)
        if not view.balance_synced:
            return None
        peak = max(view.peak_equity, view.equity)
        if peak != view.peak_equity:
            self._conn.execute("UPDATE account SET peak_equity = ? WHERE id = 1", (peak,))
        reason = None
        if self.max_drawdown_pct and view.halted_reason is None and peak > 0:
            drawdown = (peak - view.equity) / peak
            if drawdown >= self.max_drawdown_pct - 1e-12:
                reason = (
                    f"max drawdown {drawdown:.2%} from peak {peak:.2f} reached the "
                    f"{self.max_drawdown_pct:.0%} kill switch."
                )
                self._conn.execute("UPDATE account SET halted_reason = ? WHERE id = 1", (reason,))
        self._conn.commit()
        return reason

    def set_paused(self, paused: bool) -> None:
        self._conn.execute("UPDATE account SET paused = ? WHERE id = 1", (int(paused),))
        self._conn.commit()

    def reset_halt(self, marks: dict[str, float] | None = None) -> None:
        """Clear the kill switch and start a new high-water mark from current equity."""
        equity = self.view(marks).equity
        self._conn.execute(
            "UPDATE account SET halted_reason = NULL, peak_equity = ? WHERE id = 1",
            (equity,),
        )
        self._conn.commit()

    # ------------------------------------------------------------- deposits

    def apply_monthly_deposits(self, amount: float, day_of_month: int) -> list[tuple[datetime, float]]:
        """Simulated bank transfers into the paper account.

        Every month on ``day_of_month`` (00:00 UTC, clamped to the month's last
        day) ``amount`` is added as new cash. Deposits that fell due while the
        bot was not running are applied on the next cycle, each with its own
        due date. A deposit is new money, not profit: the daily-loss baseline
        and the drawdown peak move up by the same amount, so neither the daily
        PnL nor the kill switch mistakes it for a gain.
        """
        if amount <= 0:
            return []
        account = self._account()
        started = _text_to_dt(account["started_at"]) if account["started_at"] else self._now()
        now = self._now()
        due: list[datetime] = []
        year, month = started.year, started.month
        while True:
            last_day = (date_first_of_next_month(year, month) - timedelta(days=1)).day
            when = datetime(year, month, min(day_of_month, last_day), tzinfo=timezone.utc)
            if when > now:
                break
            if when > started:
                due.append(when)
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        done = self._conn.execute("SELECT COUNT(*) AS n FROM deposits WHERE note = 'monthly'").fetchone()["n"]
        applied = []
        for when in due[int(done):]:
            self._conn.execute(
                """
                UPDATE account SET cash = cash + ?, total_deposits = COALESCE(total_deposits, 0) + ?,
                    day_start_equity = day_start_equity + ?, peak_equity = peak_equity + ?
                WHERE id = 1
                """,
                (amount, amount, amount, amount),
            )
            self._conn.execute("INSERT INTO deposits (ts, amount, note) VALUES (?, ?, 'monthly')", (_dt_to_text(when), amount))
            applied.append((when, amount))
        self._conn.commit()
        return applied

    def deposits(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT ts, amount, note FROM deposits ORDER BY ts")
        return [dict(row) for row in rows]

    def total_deposits(self) -> float:
        row = self._account()
        return float(row["total_deposits"] or 0.0)

    # ----------------------------------------------------- audit and history

    def record_decision(
        self,
        signal: TechnicalSignal,
        decision: RiskDecision,
        sentiment_score: float | None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO decisions (
                ts, symbol, strategy, signal, signal_confidence, sentiment_score, action,
                confidence, price, size, stop_loss, take_profit, reasoning, indicators
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _dt_to_text(self._now()),
                decision.symbol,
                signal.strategy,
                signal.action.value,
                signal.confidence,
                sentiment_score,
                decision.action.value,
                decision.confidence,
                signal.price,
                decision.position_size,
                decision.stop_loss,
                decision.take_profit,
                decision.reasoning,
                json.dumps({k: round(v, 10) for k, v in signal.indicators.items()}),
            ),
        )
        self._conn.commit()

    def record_equity(self, marks: dict[str, float] | None = None) -> None:
        view = self.view(marks)
        self._conn.execute(
            "INSERT OR REPLACE INTO equity_snapshots (ts, equity, cash, open_positions) VALUES (?, ?, ?, ?)",
            (_dt_to_text(self._now()), view.equity, view.cash, view.open_position_count),
        )
        self._conn.commit()

    def add_event(self, level: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, level, message) VALUES (?, ?, ?)",
            (_dt_to_text(self._now()), level, message[:2000]),
        )
        self._conn.commit()

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT ts, level, message FROM events ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT ts, symbol, strategy, signal, signal_confidence, sentiment_score, action,
                   confidence, price, size, stop_loss, take_profit, reasoning
            FROM decisions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT symbol, side, size, entry, exit_price, pnl, reason, opened_at, closed_at
            FROM trades ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]

    def trade_stats(self) -> dict[str, float]:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(pnl), 0) AS pnl,
                   COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                   COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) AS gross_win,
                   COALESCE(SUM(CASE WHEN pnl < 0 THEN -pnl ELSE 0 END), 0) AS gross_loss
            FROM trades
            """
        ).fetchone()
        n = int(row["n"])
        gross_loss = float(row["gross_loss"])
        return {
            "trades": n,
            "realized_pnl": float(row["pnl"]),
            "win_rate": (int(row["wins"]) / n) if n else 0.0,
            "profit_factor": (float(row["gross_win"]) / gross_loss) if gross_loss > 0 else 0.0,
        }

    def equity_history(self, limit: int = 2000) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT ts, equity FROM (SELECT ts, equity FROM equity_snapshots ORDER BY ts DESC LIMIT ?) ORDER BY ts",
            (limit,),
        )
        return [dict(row) for row in rows]

    def first_equity(self) -> float | None:
        row = self._conn.execute("SELECT equity FROM equity_snapshots ORDER BY ts LIMIT 1").fetchone()
        return float(row["equity"]) if row else None

    def decision_counts(self, hours: float = 24) -> dict[str, int]:
        """How many decisions of each kind in the last ``hours`` (for a plain summary line)."""
        since = _dt_to_text(self._now() - timedelta(hours=hours))
        rows = self._conn.execute(
            "SELECT action, COUNT(*) AS n FROM decisions WHERE ts >= ? GROUP BY action", (since,)
        ).fetchall()
        return {row["action"]: int(row["n"]) for row in rows}

    def prune(self, retention_days: int) -> None:
        """Drop expired cooldowns and old audit rows. Trades are kept forever."""
        now = self._now()
        cutoff = _dt_to_text(now - timedelta(days=retention_days))
        self._conn.execute("DELETE FROM cooldowns WHERE until_ts <= ?", (_dt_to_text(now),))
        self._conn.execute("DELETE FROM decisions WHERE ts < ?", (cutoff,))
        self._conn.execute("DELETE FROM equity_snapshots WHERE ts < ?", (cutoff,))
        self._conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        self._conn.commit()

    def _format_day_summary(self, day: str, marks: dict[str, float] | None) -> str:
        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n, COALESCE(SUM(pnl), 0) AS pnl
            FROM trades
            WHERE closed_at >= ? AND closed_at < ?
            """,
            (_dt_to_text(start), _dt_to_text(end)),
        ).fetchone()
        view = self.view(marks)
        lines = [
            f"Daily summary {day} UTC",
            f"Closed trades: {int(row['n'])}",
            f"Realized PnL: {float(row['pnl']):.2f} {self.quote_currency}",
            f"Equity into the new day: {view.equity:.2f} {self.quote_currency}",
            f"Open positions: {view.open_position_count}",
        ]
        for position in view.positions.values():
            lines.append(
                f"- {position.side} {position.symbol} size {position.size:.8g} entry {position.entry:.8g}"
            )
        if view.open_position_count == 0:
            lines.append("- none")
        return "\n".join(lines)
