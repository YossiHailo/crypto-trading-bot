"""Historical tournament: every strategy on every coin over the last 3 years.

    python scripts/research.py                     # SYMBOLS / TIMEFRAME from the environment or .env
    python scripts/research.py --years 3 --random-runs 30 --html docs/research/index.html

Writes ``<DATA_DIR>/arena/research.json`` with a ranking the strategy tournament
uses to pick the first champion, and an HTML report with every detail.

Ranking: average Sharpe ratio across coins (return per unit of risk), so a
strategy that made a bit less with much smaller swings ranks above a lucky,
wild one. Strategies with fewer than 20 trades in total are ranked last,
because a handful of trades says almost nothing about skill.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from agents.strategies import STRATEGY_LABELS, build_strategy  # noqa: E402
from backtest.data import download_ohlcv  # noqa: E402
from backtest.engine import buy_and_hold, period_breakdown, random_entries, rows_to_frame, run_strategy, verdict  # noqa: E402
from backtest.report import SymbolReport, write_html  # noqa: E402
from config.settings import Settings  # noqa: E402
from core.timeframes import timeframe_to_ms  # noqa: E402

MIN_TRADES = 20


def _clean(value: float) -> float | None:
    value = float(value)
    return round(value, 6) if np.isfinite(value) else None


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--random-runs", type=int, default=30)
    parser.add_argument("--cache", default="history-cache")
    parser.add_argument("--html", default="", help="Also write the full HTML report here")
    args = parser.parse_args(argv)

    settings = Settings()
    names = settings.arena_list()
    symbols = settings.symbol_list()
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=365 * args.years)
    warmup = timeframe_to_ms(settings.timeframe) * (settings.ohlcv_limit + 5)
    per_strategy: dict[str, dict] = {name: {"per_symbol": {}} for name in names}
    hold_returns: dict[str, float] = {}
    reports: list[SymbolReport] = []

    for symbol in symbols:
        print(f"Downloading {symbol} {settings.timeframe} ...", flush=True)
        try:
            rows = await download_ohlcv(
                settings.exchange_id,
                symbol,
                settings.timeframe,
                int(since.timestamp() * 1000) - warmup,
                int(until.timestamp() * 1000),
                cache_dir=args.cache,
                market_type=settings.market_type,
                api_url=settings.market_data_url,
            )
        except Exception as exc:
            print(f"{symbol}: download failed: {exc}", file=sys.stderr)
            continue
        frame = rows_to_frame(rows)
        if len(frame) < settings.ohlcv_limit + 50:
            print(f"{symbol}: only {len(frame)} candles, skipped", file=sys.stderr)
            continue
        start = max(settings.ohlcv_limit - 1, max(build_strategy(n).min_bars for n in names))
        hold = buy_and_hold(frame, settings, start)
        hold_returns[symbol] = hold.metrics.total_return
        report = SymbolReport(symbol=symbol, timeframe=settings.timeframe, hold=hold, results=[])
        for name in names:
            print(f"  {symbol} {name} ...", flush=True)
            result = run_strategy(frame, settings, build_strategy(name), symbol=symbol, start_index=start, name=name)
            report.results.append(result)
            randoms = None
            if args.random_runs > 0:
                randoms = random_entries(frame, settings, len(frame) - len(result.equity), result, runs=args.random_runs)
                report.randoms[name] = randoms
            report.periods[name] = period_breakdown(result, hold, 4)
            report.verdicts[name] = verdict(result, hold, randoms, report.periods[name])
            m = result.metrics
            per_strategy[name]["per_symbol"][symbol] = {
                "total_return": _clean(m.total_return),
                "cagr": _clean(m.cagr),
                "max_drawdown": _clean(m.max_drawdown),
                "sharpe": _clean(m.sharpe),
                "trades": m.trades,
                "win_rate": _clean(m.win_rate),
                "beats_hold": bool(m.total_return > hold.metrics.total_return),
                "random_percentile": _clean(randoms.percentile_of(m.total_return)) if randoms else None,
            }
        reports.append(report)

    ranking = []
    for name, data in per_strategy.items():
        rows = list(data["per_symbol"].values())
        if not rows:
            continue

        def avg(key: str) -> float | None:
            values = [r[key] for r in rows if r[key] is not None]
            return round(float(np.mean(values)), 6) if values else None

        trades = sum(r["trades"] for r in rows)
        ranking.append(
            {
                "strategy": name,
                "label_he": STRATEGY_LABELS.get(name, {}).get("he", name),
                "label_en": STRATEGY_LABELS.get(name, {}).get("en", name),
                "avg_return": avg("total_return"),
                "avg_cagr": avg("cagr"),
                "avg_max_drawdown": avg("max_drawdown"),
                "avg_sharpe": avg("sharpe"),
                "trades": trades,
                "avg_win_rate": avg("win_rate"),
                "coins_beating_hold": sum(1 for r in rows if r["beats_hold"]),
                "coins": len(rows),
                "avg_random_percentile": avg("random_percentile"),
                "enough_trades": trades >= MIN_TRADES,
                "per_symbol": data["per_symbol"],
            }
        )
    ranking.sort(key=lambda r: (not r["enough_trades"], -(r["avg_sharpe"] or -99), -(r["avg_return"] or -99)))
    for place, row in enumerate(ranking, 1):
        row["place"] = place

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {"from": since.date().isoformat(), "to": until.date().isoformat()},
        "timeframe": settings.timeframe,
        "symbols": [r.symbol for r in reports],
        "buy_and_hold": {k: _clean(v) for k, v in hold_returns.items()},
        "ranking_rule": "average Sharpe ratio across coins; strategies with fewer than 20 trades rank last",
        "ranking": ranking,
    }
    path = Path(settings.data_dir) / "arena" / "research.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Research written to {path}")
    for row in ranking:
        print(
            f"{row['place']}. {row['strategy']:<11} sharpe {row['avg_sharpe']} return {row['avg_return']} "
            f"maxDD {row['avg_max_drawdown']} trades {row['trades']} beats hold on {row['coins_beating_hold']}/{row['coins']}"
        )
    if args.html and reports:
        summary = {k: v for k, v in settings.public_summary().items() if k not in {"mode", "sandbox"}}
        summary["strategies"] = ", ".join(names)
        html = write_html(reports, args.html, summary)
        print(f"HTML report: {html}")
    return 0 if ranking else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
