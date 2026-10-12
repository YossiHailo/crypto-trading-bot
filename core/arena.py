"""Strategy tournament ("arena"), paper trading only.

Every strategy in ``ARENA_STRATEGIES`` trades its own virtual account: the same
starting money, the same simulated monthly deposits, the same coins and the
same risk rules as the main account. They all see the same candles and the
same market mood in each cycle, so the only difference between them is the
strategy itself.

Rules
-----
* **Elimination.** When a strategy's true loss (account value minus all the
  money it was given) reaches ``ARENA_ELIMINATE_LOSS_PCT`` of that money, its
  open positions are closed and it stops trading for good.
* **Champion.** The main account trades the champion strategy when
  ``ARENA_AUTO_SELECT`` is on. The first champion is the best strategy of the
  historical backtest (``research.json``), or ``STRATEGY`` when there is none.
  Once a day a challenger replaces the champion only if it is ahead by at
  least ``ARENA_SWITCH_MARGIN`` (true return, percentage points) and has at
  least ``ARENA_MIN_TRADES`` closed trades. This keeps one lucky trade from
  flipping the main account back and forth. An eliminated champion is replaced
  at once by the best active strategy.

State lives in ``<DATA_DIR>/arena/``: one SQLite file per strategy and
``arena.json`` with statuses, standings and the switch history.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.execution import ExecutionAgent
from agents.market_data import MarketDataAgent
from agents.risk import RiskManager
from agents.strategies import STRATEGY_LABELS, build_strategy
from config.settings import Settings
from core.models import RiskAction, SentimentSignal, utcnow
from core.orchestrator import run_cycle
from core.state import PortfolioStore

logger = logging.getLogger("trading.arena")


def arena_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "arena"


def state_path(settings: Settings) -> Path:
    return arena_dir(settings) / "arena.json"


def research_path(settings: Settings) -> Path:
    return arena_dir(settings) / "research.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def research_best(settings: Settings) -> str | None:
    ranking = load_json(research_path(settings)).get("ranking") or []
    allowed = set(settings.arena_list())
    for row in ranking:
        if row.get("strategy") in allowed:
            return str(row["strategy"])
    return None


def current_champion(settings: Settings) -> str | None:
    """The strategy the main account should trade now, or None to use STRATEGY."""
    if not (settings.arena_enabled and settings.arena_auto_select and settings.paper_trading):
        return None
    champion = load_json(state_path(settings)).get("champion")
    if champion in settings.arena_list():
        return str(champion)
    return research_best(settings)


# ------------------------------------------------------------------ helpers


class CachedCandleSource:
    """Wraps a candle source so one cycle downloads each symbol's candles once."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self._cache: dict[tuple[str, str, int], list[list[float]]] = {}

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        key = (symbol, timeframe, limit)
        if key not in self._cache:
            self._cache[key] = await self.source.fetch_ohlcv(symbol, timeframe, limit)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    async def close(self) -> None:
        await self.source.close()


class _SharedSource:
    """Gives each contender the shared cache without letting it close the real client."""

    def __init__(self, cache: CachedCandleSource) -> None:
        self.cache = cache

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        return await self.cache.fetch_ohlcv(symbol, timeframe, limit)

    async def close(self) -> None:
        return None


class _FixedSentiment:
    def __init__(self, signal: SentimentSignal) -> None:
        self.signal = signal

    async def evaluate(self) -> SentimentSignal:
        return self.signal

    async def close(self) -> None:
        return None


class _Silent:
    """Notifier that drops everything: only the main account talks to Telegram."""

    def __getattr__(self, name: str):
        async def _noop(*args: object, **kwargs: object) -> bool:
            return True

        return _noop


class ArenaStore(PortfolioStore):
    """Contender account. Keeps only real actions in the decision log to stay small."""

    def record_decision(self, signal, decision, sentiment_score) -> None:  # type: ignore[override]
        if decision.action != RiskAction.NO_ACTION:
            super().record_decision(signal, decision, sentiment_score)


def _open_store(settings: Settings, name: str) -> ArenaStore:
    return ArenaStore(
        arena_dir(settings) / f"{name}.sqlite",
        settings.starting_balance,
        quote_currency=settings.quote_currency,
        fee_rate=settings.fee_rate,
        max_drawdown_pct=settings.max_drawdown_pct,
    )


def _standing(name: str, store: PortfolioStore, prices: dict[str, float], status: str) -> dict[str, Any]:
    view = store.view(prices)
    deposits = store.total_deposits() or 0.0
    stats = store.trade_stats()
    profit = view.equity - deposits
    return {
        "strategy": name,
        "label_he": STRATEGY_LABELS.get(name, {}).get("he", name),
        "label_en": STRATEGY_LABELS.get(name, {}).get("en", name),
        "status": status,
        "equity": round(view.equity, 2),
        "deposits": round(deposits, 2),
        "true_profit": round(profit, 2),
        "true_profit_pct": (profit / deposits) if deposits else 0.0,
        "trades": int(stats.get("trades", 0)),
        "wins": round(float(stats.get("win_rate", 0.0)) * int(stats.get("trades", 0))),
        "realized_pnl": round(float(stats.get("realized_pnl", 0.0)), 2),
        "open_positions": [
            {"symbol": p.symbol, "entry": p.entry, "size": p.size} for p in view.positions.values()
        ],
        "exposure_pct": (view.open_notional / view.equity) if view.equity else 0.0,
    }


# --------------------------------------------------------------------- main


async def run_arena(
    settings: Settings,
    source: CachedCandleSource,
    sentiment: SentimentSignal,
    *,
    main_store: PortfolioStore | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one cycle for every contender, apply elimination, review the champion."""
    now = now or utcnow()
    path = state_path(settings)
    state = load_json(path)
    contenders: dict[str, dict[str, Any]] = state.setdefault("contenders", {})
    history: list[dict[str, Any]] = state.setdefault("history", [])
    standings: list[dict[str, Any]] = []
    shared = _SharedSource(source)
    symbols = settings.symbol_list()

    for name in settings.arena_list():
        entry = contenders.setdefault(name, {"status": "active", "since": now.isoformat()})
        store = _open_store(settings, name)
        execution = ExecutionAgent(settings, store)
        prices: dict[str, float] = {}
        try:
            if entry.get("status") == "active":
                market = MarketDataAgent(shared, settings.timeframe, settings.ohlcv_limit, strategy=build_strategy(name))
                report = await run_cycle(
                    symbols=symbols,
                    market=market,
                    sentiment_agent=_FixedSentiment(sentiment),
                    risk=RiskManager(settings),
                    execution=execution,
                    notifier=_Silent(),
                    settings=settings,
                )
                prices = report.prices
                deposits = store.total_deposits() or 0.0
                equity = store.view(prices).equity
                if deposits and (deposits - equity) >= settings.arena_eliminate_loss_pct * deposits:
                    await execution.close_all(prices)
                    entry["status"] = "eliminated"
                    entry["eliminated_at"] = now.isoformat()
                    message = (
                        f"Arena: strategy {name} eliminated. It lost {(deposits - equity):.2f} of the "
                        f"{deposits:.2f} it was given (limit {settings.arena_eliminate_loss_pct:.0%})."
                    )
                    logger.warning(message)
                    store.add_event("warning", message)
                    if main_store is not None:
                        main_store.add_event("warning", message)
            else:
                prices = {}
            standings.append(_standing(name, store, prices, entry.get("status", "active")))
        except Exception:
            logger.exception("Arena contender %s failed", name)
            standings.append({**_standing(name, store, prices, entry.get("status", "active")), "error": True})
        finally:
            await execution.close()

    champion = _review_champion(settings, state, standings, now, main_store)
    for row in standings:
        row["champion"] = row["strategy"] == champion
    standings.sort(key=lambda r: (r["status"] != "active", -r["true_profit_pct"], -r["trades"]))
    state.update(
        {
            "updated_at": now.isoformat(),
            "champion": champion,
            "standings": standings,
            "rules": {
                "eliminate_loss_pct": settings.arena_eliminate_loss_pct,
                "switch_margin": settings.arena_switch_margin,
                "min_trades": settings.arena_min_trades,
                "auto_select": settings.arena_auto_select,
            },
        }
    )
    research = load_json(research_path(settings))
    if research:
        state["research"] = {
            "generated_at": research.get("generated_at"),
            "period": research.get("period"),
            "ranking": research.get("ranking", [])[:10],
        }
    del history[:-50]
    save_json(path, state)
    return state


def _review_champion(
    settings: Settings,
    state: dict[str, Any],
    standings: list[dict[str, Any]],
    now: datetime,
    main_store: PortfolioStore | None,
) -> str:
    by_name = {row["strategy"]: row for row in standings}
    active = [row for row in standings if row["status"] == "active"]
    champion = state.get("champion")
    if champion not in by_name:
        champion = research_best(settings) or (settings.strategy if settings.strategy in by_name else None)
        if champion is None and active:
            champion = active[0]["strategy"]
        state["champion_since"] = now.isoformat()
        state.setdefault("history", []).append(
            {"at": now.isoformat(), "to": champion, "reason": "first champion (best in the 3-year backtest)" if research_best(settings) else "first champion"}
        )

    def switch(to: str, reason: str) -> None:
        nonlocal champion
        message = f"Arena: the main account switches strategy from {champion} to {to}. {reason}"
        logger.warning(message)
        if main_store is not None:
            main_store.add_event("info", message)
        state.setdefault("history", []).append({"at": now.isoformat(), "from": champion, "to": to, "reason": reason})
        champion = to
        state["champion_since"] = now.isoformat()

    current = by_name.get(champion)
    if current is not None and current["status"] != "active" and active:
        best = max(active, key=lambda r: r["true_profit_pct"])
        switch(best["strategy"], f"{champion} was eliminated; {best['strategy']} leads the active strategies.")
    elif current is not None and active:
        today = now.date().isoformat()
        if state.get("last_review") != today:
            state["last_review"] = today
            best = max(active, key=lambda r: (r["true_profit_pct"], r["trades"]))
            lead = best["true_profit_pct"] - current["true_profit_pct"]
            if (
                best["strategy"] != champion
                and best["trades"] >= settings.arena_min_trades
                and lead >= settings.arena_switch_margin
            ):
                switch(
                    best["strategy"],
                    f"Daily review: {best['strategy']} is ahead by {lead:.1%} "
                    f"({best['true_profit_pct']:+.1%} vs {current['true_profit_pct']:+.1%}) with {best['trades']} closed trades.",
                )
    state["champion"] = champion
    return str(champion)
