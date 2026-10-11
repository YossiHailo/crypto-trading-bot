"""``python main.py backtest`` - historical test with benchmarks.

Examples:

    python main.py backtest                                   # SYMBOLS/TIMEFRAME/STRATEGY from .env, last 3 years
    python main.py backtest --strategy all --since 2021-01-01 # compare trend vs legacy
    python main.py backtest --symbols BTC/USDT --timeframe 1d --fear-greed
    python main.py backtest --csv my_candles.csv --symbols BTC/USDT

Candles are cached in data/history/, so a second run is offline and fast.
The HTML report is written to reports/ and its path is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from agents.strategies import STRATEGY_NAMES, build_strategy
from backtest.data import download_ohlcv, fear_greed_history, load_csv
from backtest.engine import buy_and_hold, period_breakdown, random_entries, rows_to_frame, run_strategy, verdict
from backtest.report import SymbolReport, console_summary, write_html
from config.settings import Settings

logger = logging.getLogger("trading.backtest")


def _date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python main.py backtest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", help="Comma-separated, default: SYMBOLS from .env")
    parser.add_argument("--timeframe", help="Default: TIMEFRAME from .env")
    parser.add_argument("--strategy", help=f"{', '.join(STRATEGY_NAMES)} or all. Default: STRATEGY from .env")
    parser.add_argument("--since", type=_date, help="YYYY-MM-DD, default: 3 years ago")
    parser.add_argument("--until", type=_date, help="YYYY-MM-DD, default: now")
    parser.add_argument("--exchange", help="binance or bybit. Default: EXCHANGE_ID from .env")
    parser.add_argument("--csv", help="Use this CSV (timestamp_ms,open,high,low,close,volume) instead of downloading")
    parser.add_argument("--random-runs", type=int, default=200, help="Random-entry benchmark runs (0 to skip)")
    parser.add_argument("--folds", type=int, default=4, help="Number of consecutive out-of-sample periods")
    parser.add_argument("--fear-greed", action="store_true", help="Apply the Fear & Greed veto using its daily history")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="reports", help="Folder for the HTML report")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    overrides: dict = {"paper_trading": True, "dashboard_enabled": False}
    if args.timeframe:
        overrides["timeframe"] = args.timeframe
    if args.exchange:
        overrides["exchange_id"] = args.exchange
    try:
        settings = Settings(**overrides)
    except ValidationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    symbols = [s.strip() for s in (args.symbols or settings.symbols).split(",") if s.strip()]
    strategy_arg = (args.strategy or settings.strategy).lower()
    names = list(STRATEGY_NAMES) if strategy_arg == "all" else [strategy_arg]
    for name in names:
        if name not in STRATEGY_NAMES:
            print(f"Unknown strategy {name!r}. Use one of {', '.join(STRATEGY_NAMES)} or all.", file=sys.stderr)
            return 1
    until = args.until or datetime.now(timezone.utc)
    since = args.since or (until - timedelta(days=3 * 365))
    if args.csv and len(symbols) != 1:
        print("--csv needs exactly one --symbols value.", file=sys.stderr)
        return 1

    fear_greed = None
    if args.fear_greed:
        try:
            fear_greed = await fear_greed_history()
            print(f"Fear & Greed history: {len(fear_greed)} days")
        except Exception as exc:
            print(f"Could not download Fear & Greed history ({exc}); continuing without it.", file=sys.stderr)

    reports: list[SymbolReport] = []
    for symbol in symbols:
        if args.csv:
            rows = load_csv(args.csv)
        else:
            # Extra warm-up history before ``since`` so the first test bar has a full window.
            from core.timeframes import timeframe_to_ms

            warmup = timeframe_to_ms(settings.timeframe) * (settings.ohlcv_limit + 5)
            print(f"Downloading {symbol} {settings.timeframe} from {settings.exchange_id} (cached in data/history)...")
            try:
                rows = await download_ohlcv(
                    settings.exchange_id,
                    symbol,
                    settings.timeframe,
                    int(since.timestamp() * 1000) - warmup,
                    int(until.timestamp() * 1000),
                    market_type=settings.market_type,
                )
            except Exception as exc:
                print(f"{symbol}: download failed: {exc}", file=sys.stderr)
                continue
        frame = rows_to_frame(rows)
        if len(frame) < settings.ohlcv_limit + 50:
            print(f"{symbol}: only {len(frame)} candles, need at least {settings.ohlcv_limit + 50}. Skipping.", file=sys.stderr)
            continue
        results = []
        for name in names:
            strategy = build_strategy(name)
            start = max(settings.ohlcv_limit - 1, strategy.min_bars)
            print(f"Running {name} on {symbol} ({len(frame) - start} bars)...")
            results.append(run_strategy(frame, settings, strategy, symbol=symbol, fear_greed=fear_greed, start_index=start, name=name))
        start = max(settings.ohlcv_limit - 1, max(build_strategy(n).min_bars for n in names))
        hold = buy_and_hold(frame, settings, start)
        report = SymbolReport(symbol=symbol, timeframe=settings.timeframe, hold=hold, results=results)
        for result in results:
            randoms = None
            if args.random_runs > 0:
                print(f"Random-entry benchmark for {result.name}: {args.random_runs} runs...")
                randoms = random_entries(frame, settings, len(frame) - len(result.equity), result, runs=args.random_runs, seed=args.seed)
                report.randoms[result.name] = randoms
            report.periods[result.name] = period_breakdown(result, hold, args.folds)
            report.verdicts[result.name] = verdict(result, hold, randoms, report.periods[result.name])
        print(console_summary(report))
        reports.append(report)

    if not reports:
        print("No symbol could be tested.", file=sys.stderr)
        return 1
    summary = {k: v for k, v in settings.public_summary().items() if k not in {"mode", "symbols", "sandbox"}}
    summary["strategies"] = ", ".join(names)
    summary["fear_greed_veto"] = bool(fear_greed)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = write_html(reports, f"{args.out}/backtest-{stamp}.html", summary)
    print(f"\nHTML report: {path.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
