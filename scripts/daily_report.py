"""Daily activity report from the bot's SQLite file. Standard library only.

    python scripts/daily_report.py                       # today (Israel time) so far
    python scripts/daily_report.py --date 2026-10-08     # a given local day
    python scripts/daily_report.py --json                # machine-readable output only

Writes reports/daily/<date>.json and reports/daily/<date>.md next to the bot.
Reads a *copy* of the database, so it is safe while the bot is running.

What it measures for the day:
* uptime: how many 15-minute checks ran out of those expected, and the gaps
  (computer asleep, off or offline),
* account: value at the start and end of the day (deposits excluded from the
  change), true profit since start (value minus all money deposited), low and high, drawdown,
* trades opened and closed, with PnL and the reason each one closed,
* every decision, grouped by the reason the bot did or did not trade,
* per coin: the price move the bot saw, trend state, and how close the price
  came to the buy trigger,
* sentiment range, warnings and errors,
* running totals since the bot started.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from zoneinfo import ZoneInfo

    def _tz(name: str):
        try:
            return ZoneInfo(name)
        except Exception:
            return timezone(timedelta(hours=3))
except ImportError:  # pragma: no cover

    def _tz(name: str):
        return timezone(timedelta(hours=3))


REASONS = [
    (r"^Execute long", "buy", "קנייה"),
    (r"^Execute short", "short", "מכירה בחסר"),
    (r"^Exit signal closes", "exit", "יציאה לפי אות"),
    (r"Technical signal is HOLD", "no_signal", "אין אות"),
    (r"already open", "already_open", "כבר יש עסקה פתוחה"),
    (r"limit order is already resting", "pending", "פקודה ממתינה"),
    (r"shorting is disabled", "short_disabled", "אות ירידה, מכירה בחסר כבויה"),
    (r"require MARKET_TYPE=swap", "short_disabled", "אות ירידה, מכירה בחסר לא אפשרית"),
    (r"paused by the operator", "paused", "השהיה ידנית"),
    (r"Trading halted", "halted", "עצירת חירום"),
    (r"has not synced", "not_synced", "ממתין ליתרה"),
    (r"circuit breaker", "daily_limit", "מגבלת הפסד יומית"),
    (r"Cooldown", "cooldown", "המתנה אחרי סטופ"),
    (r"Max open positions", "max_positions", "מקסימום עסקאות"),
    (r"Sentiment veto", "sentiment_veto", "וטו של סנטימנט"),
    (r"Combined confidence", "low_confidence", "אות חלש"),
    (r"Reward:risk", "low_rr", "רווח/סיכון נמוך"),
    (r"Position size is zero", "size_zero", "סכום קטן מדי"),
]


def classify(reasoning: str) -> tuple[str, str]:
    for pattern, key, label in REASONS:
        if re.search(pattern, reasoning or ""):
            return key, label
    return "other", "אחר"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def open_copy(db: Path) -> sqlite3.Connection:
    """Copy the database (and its WAL) to a temp dir so the live file is never locked."""
    tmp = Path(tempfile.mkdtemp(prefix="ctb-report-"))
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(db) + suffix)
        if src.exists():
            shutil.copy2(src, tmp / (db.name + suffix))
    conn = sqlite3.connect(tmp / db.name)
    conn.row_factory = sqlite3.Row
    return conn


def build(conn: sqlite3.Connection, day: date, tzname: str, interval_min: float, now: datetime) -> dict:
    tz = _tz(tzname)
    start_local = datetime.combine(day, datetime.min.time(), tz)
    end_local = start_local + timedelta(days=1)
    start, end = start_local.astimezone(timezone.utc), min(end_local.astimezone(timezone.utc), now)
    a, b = _iso(start), _iso(end)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    # ---------------------------------------------------------------- uptime
    snaps = [dict(r) for r in conn.execute("SELECT ts, equity, cash, open_positions FROM equity_snapshots ORDER BY ts")] if "equity_snapshots" in tables else []
    day_snaps = [s for s in snaps if a <= s["ts"] < b]
    window_min = max(0.0, (end - start).total_seconds() / 60)
    expected = int(window_min // interval_min)
    gaps = []
    prev = start
    for s in day_snaps:
        t = _parse(s["ts"])
        if (t - prev).total_seconds() > interval_min * 60 * 2 + 120:
            gaps.append({"from": _iso(prev), "to": _iso(t), "minutes": round((t - prev).total_seconds() / 60)})
        prev = t
    if (end - prev).total_seconds() > interval_min * 60 * 2 + 120:
        gaps.append({"from": _iso(prev), "to": _iso(end), "minutes": round((end - prev).total_seconds() / 60), "ongoing": True})
    downtime = sum(g["minutes"] for g in gaps)
    uptime = {
        "checks_ran": len(day_snaps),
        "checks_expected": expected,
        "coverage_pct": round(100 * min(1.0, len(day_snaps) / expected), 1) if expected else None,
        "downtime_minutes": downtime,
        "gaps": gaps,
        "last_check": day_snaps[-1]["ts"] if day_snaps else (snaps[-1]["ts"] if snaps else None),
    }

    # --------------------------------------------------------------- account
    before = [s for s in snaps if s["ts"] < a]
    start_equity = before[-1]["equity"] if before else (day_snaps[0]["equity"] if day_snaps else None)
    end_equity = day_snaps[-1]["equity"] if day_snaps else start_equity
    # Deposits (starting balance + simulated monthly bank transfers) are money put
    # in, not profit: they are removed from the day's change and from "since start".
    deposits = [dict(r) for r in conn.execute("SELECT ts, amount, note FROM deposits ORDER BY ts")] if "deposits" in tables else []
    deposited_today = round(sum(d["amount"] for d in deposits if d["note"] != "initial" and a <= d["ts"] < b), 2)
    deposited_total = sum(d["amount"] for d in deposits if d["ts"] < b or d["note"] == "initial")
    baseline = deposited_total or (snaps[0]["equity"] if snaps else None)
    values = [s["equity"] for s in day_snaps]
    peak_all = max((s["equity"] for s in snaps if s["ts"] < b), default=None)
    change = (end_equity - start_equity - deposited_today) if start_equity is not None and end_equity is not None else None
    account = {
        "start": start_equity,
        "end": end_equity,
        "deposited_today": deposited_today,
        "deposited_total": baseline,
        "change": change,
        "change_pct": (change / start_equity * 100) if start_equity and change is not None else None,
        "low": min(values) if values else None,
        "high": max(values) if values else None,
        "baseline": baseline,
        "since_start": (end_equity - baseline) if baseline and end_equity is not None else None,
        "since_start_pct": ((end_equity / baseline - 1) * 100) if baseline and end_equity else None,
        "drawdown_from_peak_pct": ((peak_all - end_equity) / peak_all * 100) if peak_all and end_equity is not None else None,
        "cash_end": day_snaps[-1]["cash"] if day_snaps else None,
    }
    acct_row = conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
    if acct_row is not None:
        keys = acct_row.keys()
        account["paused"] = bool(acct_row["paused"]) if "paused" in keys else False
        account["halted_reason"] = acct_row["halted_reason"] if "halted_reason" in keys else None

    # ---------------------------------------------------------------- trades
    closed = [dict(r) for r in conn.execute(
        "SELECT symbol, side, size, entry, exit_price, pnl, reason, opened_at, closed_at FROM trades WHERE closed_at >= ? AND closed_at < ? ORDER BY closed_at", (a, b))]
    opened_closed = [dict(r) for r in conn.execute(
        "SELECT symbol, side, size, entry, opened_at FROM trades WHERE opened_at >= ? AND opened_at < ?", (a, b))]
    positions = [dict(r) for r in conn.execute("SELECT symbol, side, size, entry, stop_loss, take_profit, opened_at, exchange_protected FROM positions")]
    opened_today = opened_closed + [p for p in positions if a <= p["opened_at"] < b]
    for t in closed:
        t["hold_hours"] = round((_parse(t["closed_at"]) - _parse(t["opened_at"])).total_seconds() / 3600, 1)
        t["return_pct"] = round(((t["exit_price"] - t["entry"]) if t["side"] == "long" else (t["entry"] - t["exit_price"])) / t["entry"] * 100, 2)
    trades = {
        "opened": len(opened_today),
        "closed": len(closed),
        "wins": sum(1 for t in closed if t["pnl"] > 0),
        "pnl": round(sum(t["pnl"] for t in closed), 2),
        "closed_list": closed,
        "opened_list": opened_today,
        "open_at_end": positions,
    }

    # ------------------------------------------------- decisions and coins
    decisions = [dict(r) for r in conn.execute(
        "SELECT ts, symbol, strategy, signal, signal_confidence, sentiment_score, action, price, reasoning, indicators FROM decisions WHERE ts >= ? AND ts < ? ORDER BY ts", (a, b))] if "decisions" in tables else []
    reasons = Counter()
    per_coin: dict[str, dict] = defaultdict(lambda: {"signals": Counter(), "prices": [], "closest_to_buy_pct": None, "exit_flags": 0})
    sentiments = []
    for d in decisions:
        key, label = classify(d["reasoning"])
        reasons[label] += 1
        coin = per_coin[d["symbol"]]
        coin["signals"][d["signal"]] += 1
        if d["price"]:
            coin["prices"].append((d["ts"], d["price"]))
        if d["sentiment_score"] is not None:
            sentiments.append(d["sentiment_score"])
        try:
            ind = json.loads(d["indicators"] or "{}")
        except ValueError:
            ind = {}
        coin["last_indicators"] = ind
        coin["last_reasoning"] = d["reasoning"]
        if ind.get("upper") and d["price"]:
            dist = (ind["upper"] - d["price"]) / d["price"] * 100
            uptrend = ind.get("ema_fast", 0) > ind.get("ema_slow", 0) and ind.get("close", 0) > ind.get("ema_slow", 0)
            if uptrend and (coin["closest_to_buy_pct"] is None or dist < coin["closest_to_buy_pct"]):
                coin["closest_to_buy_pct"] = round(dist, 2)
        if "exit lon" in (d["reasoning"] or ""):
            coin["exit_flags"] += 1
    coins = {}
    for symbol, c in per_coin.items():
        prices = [p for _, p in c["prices"]]
        ind = c.get("last_indicators", {})
        if ind.get("ema_fast") is not None and ind.get("ema_slow") is not None and ind.get("close") is not None:
            trend = "up" if ind["ema_fast"] > ind["ema_slow"] and ind["close"] > ind["ema_slow"] else "down" if ind["ema_fast"] < ind["ema_slow"] and ind["close"] < ind["ema_slow"] else "mixed"
        else:
            trend = "unknown"
        coins[symbol] = {
            "checks": sum(c["signals"].values()),
            "signals": dict(c["signals"]),
            "price_first": prices[0] if prices else None,
            "price_last": prices[-1] if prices else None,
            "price_low": min(prices) if prices else None,
            "price_high": max(prices) if prices else None,
            "move_pct": round((prices[-1] / prices[0] - 1) * 100, 2) if len(prices) > 1 else None,
            "trend_at_end": trend,
            "buy_trigger_at_end": ind.get("upper"),
            "exit_level_at_end": ind.get("exit_lower"),
            "closest_to_buy_trigger_pct": c["closest_to_buy_pct"],
            "exit_signals": c["exit_flags"],
            "last_reasoning": c.get("last_reasoning"),
        }

    # ------------------------------------------------------------ events
    events = [dict(r) for r in conn.execute("SELECT ts, level, message FROM events WHERE ts >= ? AND ts < ? ORDER BY ts", (a, b))] if "events" in tables else []
    warnings = [e for e in events if e["level"] in {"warning", "error"}]

    # ------------------------------------------------------- since start
    all_trades = [dict(r) for r in conn.execute("SELECT pnl, closed_at FROM trades ORDER BY closed_at")]
    first_ts = snaps[0]["ts"] if snaps else None
    cumulative = {
        "running_since": first_ts,
        "days_running": round((end - _parse(first_ts)).total_seconds() / 86400, 1) if first_ts else 0,
        "trades": len(all_trades),
        "wins": sum(1 for t in all_trades if t["pnl"] > 0),
        "pnl": round(sum(t["pnl"] for t in all_trades), 2),
        "checks_total": len([s for s in snaps if s["ts"] < b]),
    }

    return {
        "date": day.isoformat(),
        "timezone": tzname,
        "window": {"from_utc": a, "to_utc": b, "complete_day": end >= end_local.astimezone(timezone.utc)},
        "uptime": uptime,
        "account": account,
        "trades": trades,
        "decisions": {"total": len(decisions), "by_reason": dict(reasons.most_common())},
        "coins": coins,
        "sentiment": {
            "min": min(sentiments) if sentiments else None,
            "max": max(sentiments) if sentiments else None,
            "avg": round(sum(sentiments) / len(sentiments), 3) if sentiments else None,
        },
        "warnings": warnings[-40:],
        "warning_count": len(warnings),
        "cumulative": cumulative,
    }


def _money(v) -> str:
    return "—" if v is None else f"${v:,.2f}"


def _pct(v) -> str:
    return "—" if v is None else f"{v:+.2f}%"


def to_markdown(r: dict) -> str:
    acc, up, tr = r["account"], r["uptime"], r["trades"]
    lines = [
        f"# דוח יומי {r['date']}",
        "",
        f"- **שווי החשבון:** {_money(acc['start'])} → {_money(acc['end'])}. שינוי מהמסחר: {_money(acc['change'])} ({_pct(acc['change_pct'])})"
        + (f", בנוסף הפקדה חודשית {_money(acc['deposited_today'])}" if acc.get("deposited_today") else "") + ".",
        f"- **רווח אמיתי מאז ההתחלה:** {_money(acc['since_start'])} ({_pct(acc['since_start_pct'])}) על {_money(acc.get('deposited_total'))} שהופקדו בסך הכול.",
        f"- **זמן פעילות:** {up['checks_ran']} בדיקות מתוך {up['checks_expected']} צפויות ({up['coverage_pct']}%). זמן השבתה: {up['downtime_minutes']} דקות.",
        f"- **עסקאות:** נפתחו {tr['opened']}, נסגרו {tr['closed']} ({tr['wins']} ברווח), רווח/הפסד ממומש {_money(tr['pnl'])}.",
        f"- **החלטות:** {r['decisions']['total']}. " + ", ".join(f"{k}: {v}" for k, v in r["decisions"]["by_reason"].items()),
        "",
        "## מטבעות",
        "",
        "| מטבע | שינוי מחיר | מגמה בסוף היום | הכי קרוב לטריגר קנייה | אותות |",
        "| --- | --- | --- | --- | --- |",
    ]
    names = {"up": "עולה", "down": "יורדת", "mixed": "לא ברורה", "unknown": "—"}
    for sym, c in r["coins"].items():
        close = "—" if c["closest_to_buy_trigger_pct"] is None else f"{c['closest_to_buy_trigger_pct']:.2f}%"
        lines.append(f"| {sym} | {_pct(c['move_pct'])} | {names.get(c['trend_at_end'], c['trend_at_end'])} | {close} | {c['signals']} |")
    if up["gaps"]:
        lines += ["", "## השבתות", ""] + [f"- {g['from']} → {g['to']} ({g['minutes']} דק')" for g in up["gaps"]]
    if r["warnings"]:
        lines += ["", "## אזהרות", ""] + [f"- {w['ts']}: {w['message']}" for w in r["warnings"][-15:]]
    c = r["cumulative"]
    lines += ["", f"מצטבר: {c['days_running']} ימים, {c['trades']} עסקאות, {c['wins']} ברווח, {_money(c['pnl'])}."]
    return "\n".join(lines) + "\n"


_PAGE = """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#f6f5f2; --fg:#1d1d1b; --muted:#6b6a66; --line:#e2e0da; --card:#fff; --accent:#1f6feb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --fg:#ecebe7; --muted:#a3a19b; --line:#2f2e2b; --card:#1d1d1b; --accent:#58a6ff; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.65 system-ui, "Segoe UI", Arial, sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:26px; margin:0 0 16px; }} h2 {{ font-size:19px; margin:28px 0 10px; }}
a {{ color:var(--accent); }} ul {{ padding-inline-start:22px; }} li {{ margin:4px 0; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 20px; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:14.5px; }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:start; white-space:nowrap; }}
.muted {{ color:var(--muted); font-size:14px; }}
</style></head><body><main>{body}</main></body></html>
"""


def _inline(text: str) -> str:
    import html
    import re

    out = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)


def markdown_to_html(md: str, title: str, back: str | None = "index.html") -> str:
    """Enough Markdown for the daily report: headings, bullets, tables, paragraphs."""
    parts: list[str] = []
    if back:
        parts.append(f'<p class="muted"><a href="../">← לדשבורד</a> · <a href="{back}">כל הדוחות</a></p>')
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            parts.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            parts.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            parts.append('<div class="card"><ul>' + "".join(items) + "</ul></div>")
            continue
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= {"-", " ", ":"} for c in cells):
                    tag = "th" if not rows else "td"
                    rows.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
                i += 1
            parts.append('<div class="wrap card"><table>' + "".join(rows) + "</table></div>")
            continue
        elif line.strip():
            parts.append(f"<p>{_inline(line)}</p>")
        i += 1
    return _PAGE.format(title=title, body="\n".join(parts))


def write_index(out: Path) -> None:
    """reports/index.html: newest report first."""
    days = sorted((p.stem for p in out.glob("????-??-??.html")), reverse=True)
    items = "".join(f'<li><a href="{d}.html">{d}</a></li>' for d in days) or "<li>עדיין אין דוחות.</li>"
    body = ('<p class="muted"><a href="../">← לדשבורד</a></p><h1>דוחות יומיים</h1>'
            '<p class="muted">דוח אוטומטי לכל יום (שעון ישראל). הדוח של היום מתעדכן אחרי כל בדיקה.</p>'
            f'<div class="card"><ul>{items}</ul></div>')
    (out / "index.html").write_text(_PAGE.format(title="דוחות יומיים", body=body), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="Local date YYYY-MM-DD (default: today in --tz)")
    parser.add_argument("--tz", default="Asia/Jerusalem")
    parser.add_argument("--db", default=str(ROOT / "data" / "portfolio.sqlite"))
    parser.add_argument("--interval", type=float, default=15.0, help="Minutes between checks")
    parser.add_argument("--out", default=str(ROOT / "reports" / "daily"))
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    day = date.fromisoformat(args.date) if args.date else now.astimezone(_tz(args.tz)).date()
    db = Path(args.db)
    if not db.exists():
        print(f"Database not found: {db}", file=sys.stderr)
        return 1
    report = build(open_copy(db), day, args.tz, args.interval, now)
    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{day.isoformat()}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out / f"{day.isoformat()}.md").write_text(to_markdown(report), encoding="utf-8")
        (out / f"{day.isoformat()}.html").write_text(markdown_to_html(to_markdown(report), f"דוח יומי {day.isoformat()}"), encoding="utf-8")
        write_index(out)
    except OSError as exc:
        print(f"Could not write report files: {exc}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
