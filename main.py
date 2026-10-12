"""Entry point.

    python main.py                 run the bot until Ctrl+C (dashboard on http://127.0.0.1:8765)
    python main.py --once          one cycle, then exit
    python main.py --once --action pause --export-state site/state.json
                                   one cycle for a scheduler (GitHub Actions), with a
                                   control action and a JSON snapshot for a static page
    python main.py backtest ...    historical backtest with benchmarks (see --help)

Paper trading is the default. One symbol or one agent failing is logged and
skipped; it does not stop the loop. This software can lose money. Nothing
here is financial advice.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from agents.execution import ExecutionAgent
from agents.market_data import CcxtCandleSource, FallbackCandleSource, MarketDataAgent
from agents.risk import RiskManager
from agents.sentiment import SentimentAgent
from agents.strategies import build_strategy
from config.settings import Settings, assert_trading_mode
from core.arena import CachedCandleSource, current_champion, run_arena
from core.control import Controller
from core.errors import LiveTradingNotConfirmedError
from core.exchange import build_ccxt_exchange, build_market_data_exchanges
from core.logging_config import setup_logging
from core.models import utcnow
from core.orchestrator import run_cycle
from core.redact import register_secrets
from core.state import PortfolioStore
from notifications.telegram import TelegramNotifier

logger = logging.getLogger("trading.main")

# Exit codes start_bot.bat understands: 0 = stopped on purpose, 2/3 = configuration
# problem (restarting would not help). Anything else is a crash and is restarted.
EXIT_LIVE_REFUSED = 2
EXIT_CONFIG = 3


def keep_system_awake() -> bool:
    """Ask Windows not to sleep while the bot runs (screen may still turn off).

    Without this a laptop on default power settings sleeps after a few minutes
    idle and the bot silently stops checking prices. Closing the lid or
    choosing Sleep by hand still wins.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required))
    except Exception:
        logger.warning("Could not ask Windows to stay awake", exc_info=True)
        return False


def _record_gap(store: PortfolioStore, since: datetime | None, until: datetime, interval_s: float) -> None:
    """Log an event when no cycle ran for a while (computer asleep, off or offline)."""
    if since is None:
        return
    gap = (until - since).total_seconds()
    if gap > interval_s * 2 + 120:
        minutes = int(gap // 60)
        message = (
            f"Downtime: no checks from {since.isoformat(timespec='minutes')} "
            f"to {until.isoformat(timespec='minutes')} ({minutes} min). "
            "The bot was not running (computer asleep, off or offline, or a delayed scheduled run)."
        )
        logger.warning(message)
        store.add_event("warning", message)


def _last_heartbeat(data_dir: str) -> datetime | None:
    try:
        text = (Path(data_dir) / "heartbeat").read_text(encoding="utf-8").strip()
        stamp = datetime.fromisoformat(text)
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (OSError, ValueError):
        return None


class Runtime:
    """Wired agents for one process. ``close`` is safe to call more than once."""

    def __init__(
        self,
        settings: Settings,
        store: PortfolioStore,
        market: MarketDataAgent,
        sentiment: SentimentAgent,
        risk: RiskManager,
        execution: ExecutionAgent,
        notifier: TelegramNotifier,
    ) -> None:
        self.settings = settings
        self.store = store
        self.market = market
        self.sentiment = sentiment
        self.risk = risk
        self.execution = execution
        self.notifier = notifier
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, closer in (
            ("market", self.market.close),
            ("sentiment", self.sentiment.close),
            ("execution", self.execution.close),
            ("notifier", self.notifier.close),
        ):
            try:
                await closer()
            except Exception:
                logger.exception("Error while closing %s", name)


def active_strategy(settings: Settings) -> str:
    """STRATEGY, or the tournament champion when ARENA_AUTO_SELECT is on."""
    return current_champion(settings) or settings.strategy


def build_runtime(settings: Settings) -> Runtime:
    """Construct agents. Live mode has already been confirmed by the caller."""
    store = PortfolioStore(
        f"{settings.data_dir}/portfolio.sqlite",
        settings.starting_balance,
        quote_currency=settings.quote_currency,
        fee_rate=settings.fee_rate,
        exchange_balance_tracking=not settings.paper_trading,
        market_type=settings.market_type,
        leverage=settings.leverage,
        max_drawdown_pct=settings.max_drawdown_pct,
    )
    sources = [CcxtCandleSource(client, settings.request_timeout_seconds) for client in build_market_data_exchanges(settings)]
    # One download per symbol per cycle, shared by the main account and the strategy tournament.
    candles = CachedCandleSource(sources[0] if len(sources) == 1 else FallbackCandleSource(sources))
    market = MarketDataAgent(
        candles,
        settings.timeframe,
        settings.ohlcv_limit,
        strategy=build_strategy(active_strategy(settings)),
    )
    sentiment = SentimentAgent(settings)
    risk = RiskManager(settings)
    live_exchange = None
    if not settings.paper_trading:
        live_exchange = build_ccxt_exchange(settings, authenticated=True)
    execution = ExecutionAgent(settings, store, live_exchange=live_exchange)
    notifier = TelegramNotifier(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return Runtime(settings, store, market, sentiment, risk, execution, notifier)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous crypto trading bot (paper by default)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not start the local web dashboard")
    parser.add_argument(
        "--action",
        choices=sorted(ONCE_ACTIONS),
        default="none",
        help="With --once: apply a control action (pause/resume before the cycle, reset-halt/close-all after it)",
    )
    parser.add_argument("--export-state", metavar="PATH", help="With --once: write the dashboard JSON to PATH after the cycle")
    args = parser.parse_args(argv)
    if (args.action != "none" or args.export_state) and not args.once:
        parser.error("--action and --export-state need --once")
    return args


ONCE_ACTIONS = {"none", "pause", "resume", "reset-halt", "close-all"}


async def _apply_once_action(runtime: Runtime, action: str, prices: dict[str, float]) -> None:
    """Control actions for scheduled single-cycle runs (no dashboard process to click)."""
    store = runtime.store
    if action == "pause":
        store.set_paused(True)
        message = "New entries paused (remote control). Stops and exits keep running."
    elif action == "resume":
        store.set_paused(False)
        message = "New entries resumed (remote control)."
    elif action == "reset-halt":
        store.reset_halt(prices)
        message = "Kill switch reset (remote control). The equity peak starts again from now."
    elif action == "close-all":
        results = await runtime.execution.close_all(prices)
        closed = sum(1 for result in results if result.closed_trade is not None)
        failed = [result.message for result in results if not result.success]
        message = f"Close-all (remote control): {closed} position(s) closed."
        if failed:
            message += " Failures: " + "; ".join(failed)
    else:
        return
    logger.warning(message)
    store.add_event("warning" if action == "close-all" else "info", message)
    await runtime.notifier.info(message)


async def _run_arena_step(runtime: Runtime, report: object) -> None:
    """Strategy tournament after the main cycle. Never stops the main bot."""
    from core.models import neutral_sentiment

    candles = getattr(runtime.market, "source", None)
    if not isinstance(candles, CachedCandleSource):
        return
    sentiment = getattr(report, "sentiment", None) or neutral_sentiment("Arena: no sentiment this cycle.")
    try:
        state = await run_arena(runtime.settings, candles, sentiment, main_store=runtime.store)
    except Exception:
        logger.exception("Strategy tournament failed; the main account is not affected")
        return
    champion = state.get("champion")
    if runtime.settings.arena_auto_select and champion and champion != getattr(runtime.market.strategy, "name", None):
        runtime.market.strategy = build_strategy(champion)
        logger.warning("Main account now trades the %s strategy (tournament champion).", champion)


def _export_state(runtime: Runtime, controller: Controller, path: str) -> None:
    import json

    from dashboard.server import DashboardContext, build_state, json_default

    state = build_state(DashboardContext(runtime.settings, runtime.store, runtime.execution, controller))
    state["static"] = True
    state["generated_at"] = utcnow()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(state, default=json_default, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    logger.info("Dashboard snapshot written to %s", target)


async def _start_dashboard(runtime: Runtime, controller: Controller) -> object | None:
    from dashboard.server import DashboardContext, DashboardServer

    server = DashboardServer(
        DashboardContext(
            settings=runtime.settings,
            store=runtime.store,
            execution=runtime.execution,
            controller=controller,
            notifier=runtime.notifier,
        )
    )
    try:
        await server.start()
    except OSError as exc:
        logger.error("Dashboard could not start on port %s: %s", runtime.settings.dashboard_port, exc)
        return None
    host = runtime.settings.dashboard_host
    shown = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    logger.info("Dashboard: http://%s:%s", shown, server.port)
    return server


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as exc:
        print("Configuration error. Fix .env and start again:\n", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error.get("loc", ())) or "settings"
            print(f"  - {field.upper()}: {error.get('msg')}", file=sys.stderr)
        return EXIT_CONFIG

    register_secrets(settings.secret_values())
    log_path = setup_logging(settings.log_level, settings.log_dir)
    try:
        mode = assert_trading_mode(settings)
    except LiveTradingNotConfirmedError as exc:
        logger.error("%s", exc)
        return EXIT_LIVE_REFUSED

    logger.info("Log file: %s", log_path)
    logger.warning("This bot can lose money. Mode=%s. Not financial advice.", mode)
    if mode == "live":
        logger.warning(
            "LIVE TRADING ENABLED on %s (%s%s). Real orders will be sent. Data dir: %s",
            settings.exchange_id,
            settings.market_type,
            ", sandbox" if settings.exchange_sandbox else "",
            settings.data_dir,
        )
    else:
        logger.info("Paper trading is ON. No real orders will be sent.")
    logger.info(
        "Strategy=%s symbols=%s timeframe=%s interval=%smin risk_per_trade=%.2f%% R:R=%.2f",
        active_strategy(settings),
        ",".join(settings.symbol_list()),
        settings.timeframe,
        settings.loop_interval_minutes,
        settings.risk_per_trade * 100,
        settings.reward_risk_ratio,
    )

    runtime = build_runtime(settings)
    controller = Controller(mode)
    _install_signal_handlers(controller)
    dashboard = None
    try:
        if settings.dashboard_enabled and not args.no_dashboard and not args.once:
            dashboard = await _start_dashboard(runtime, controller)
        try:
            findings = await runtime.execution.reconcile(settings.symbol_list())
        except Exception as exc:
            findings = [f"Startup reconciliation failed: {exc}"]
        for note in findings:
            logger.warning("Reconcile: %s", note)
            runtime.store.add_event("warning", f"Startup: {note}")
            await runtime.notifier.error(f"Startup reconciliation: {note}")
        if not args.once:  # a scheduler runs --once every few minutes; do not spam
            await runtime.notifier.info(
                f"Trading bot started in {mode} mode on {settings.exchange_id}. "
                f"Strategy {settings.strategy}. Symbols: {', '.join(settings.symbol_list())}."
            )
            runtime.store.add_event("info", f"Bot started in {mode} mode.")
        if keep_system_awake():
            logger.info("Windows sleep is blocked while the bot runs.")
        interval_s = settings.loop_interval_minutes * 60
        last_cycle = _last_heartbeat(settings.data_dir)
        if args.action in {"pause", "resume"}:
            await _apply_once_action(runtime, args.action, {})
        while not controller.stop_event.is_set():
            _record_gap(runtime.store, last_cycle, utcnow(), interval_s)
            quick_retry = False
            controller.wake_event.clear()
            controller.status.cycle_running = True
            candles = getattr(runtime.market, "source", None)
            if isinstance(candles, CachedCandleSource):
                candles.clear()  # fresh candles every cycle
            try:
                async with controller.cycle_lock:
                    report = await run_cycle(
                        symbols=settings.symbol_list(),
                        market=runtime.market,
                        sentiment_agent=runtime.sentiment,
                        risk=runtime.risk,
                        execution=runtime.execution,
                        notifier=runtime.notifier,
                        settings=settings,
                        stop=controller.stop_event,
                    )
                controller.record_report(report, report.prices)
                if settings.arena_enabled and settings.paper_trading:
                    await _run_arena_step(runtime, report)
                last_cycle = report.finished_at or utcnow()
                # No prices at all (typically right after the computer wakes up and the
                # network is not back yet): try again in a minute instead of waiting a full interval.
                quick_retry = not report.signals and bool(report.errors)
            except Exception:
                logger.exception("Cycle crashed unexpectedly")
                await runtime.notifier.error("Cycle crashed unexpectedly. See the log.")
            finally:
                controller.status.cycle_running = False
            if args.once:
                prices = dict(controller.status.last_prices)
                if args.action in {"reset-halt", "close-all"}:
                    async with controller.cycle_lock:
                        await _apply_once_action(runtime, args.action, prices)
                if args.export_state:
                    _export_state(runtime, controller, args.export_state)
                break
            if controller.stop_event.is_set():
                break
            delay = 60.0 if quick_retry else interval_s
            controller.status.next_cycle_at = utcnow() + timedelta(seconds=delay)
            try:
                # Wakes early on "run now" from the dashboard or on a stop signal.
                await asyncio.wait_for(controller.wake_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            controller.status.next_cycle_at = None
    finally:
        logger.info("Shutting down")
        if dashboard is not None:
            await dashboard.close()  # type: ignore[attr-defined]
        await runtime.close()
    return 0


def _install_signal_handlers(controller: Controller) -> None:
    loop = asyncio.get_running_loop()

    def _stop(signame: str) -> None:
        logger.info("Received %s, finishing the current step and shutting down", signame)
        controller.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop, sig.name)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_args, name=sig.name: loop.call_soon_threadsafe(_stop, name))


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "backtest":
        from backtest.cli import main as backtest_main

        raise SystemExit(backtest_main(argv[1:]))
    try:
        code = asyncio.run(amain(argv))
    except KeyboardInterrupt:
        code = 0
    raise SystemExit(code)


if __name__ == "__main__":
    main()
