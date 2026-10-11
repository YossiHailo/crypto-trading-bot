"""Console summary and a self-contained HTML report (inline SVG, no external files)."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from backtest.engine import BacktestResult, Metrics, PeriodRow, RandomBenchmark

# Categorical slots 1-3 of the validated reference palette (light / dark).
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70")]


@dataclass
class SymbolReport:
    symbol: str
    timeframe: str
    hold: BacktestResult
    results: list[BacktestResult]
    randoms: dict[str, RandomBenchmark] = field(default_factory=dict)
    periods: dict[str, list[PeriodRow]] = field(default_factory=dict)
    verdicts: dict[str, list[str]] = field(default_factory=dict)


def _pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:+.1f}%"


def _num(value: float, digits: int = 2) -> str:
    if value == float("inf"):
        return "∞"
    return f"{value:.{digits}f}"


METRIC_ROWS: list[tuple[str, str, callable]] = [
    ("total_return", "Total return · תשואה כוללת", _pct),
    ("cagr", "CAGR · תשואה שנתית", _pct),
    ("max_drawdown", "Max drawdown · ירידה מקסימלית", lambda v: f"{v * 100:.1f}%"),
    ("sharpe", "Sharpe", _num),
    ("sortino", "Sortino", _num),
    ("calmar", "Calmar", _num),
    ("trades", "Trades · עסקאות", lambda v: str(int(v))),
    ("win_rate", "Win rate · אחוז הצלחה", lambda v: f"{v * 100:.0f}%"),
    ("profit_factor", "Profit factor", _num),
    ("expectancy_r", "Expectancy (R) · תוחלת", lambda v: f"{v:+.2f}R"),
    ("avg_bars_held", "Avg bars held · משך ממוצע", lambda v: f"{v:.1f}"),
    ("exposure", "Time in market · זמן בשוק", lambda v: f"{v * 100:.0f}%"),
    ("fees", "Fees paid · עמלות", lambda v: f"{v:,.0f}"),
]


# Per-trade statistics mean nothing for a single buy-and-hold "trade".
_NOT_FOR_HOLD = {"trades", "win_rate", "profit_factor", "expectancy_r", "avg_bars_held"}


def _cell(key: str, fmt, metrics: Metrics, is_hold: bool) -> str:
    if is_hold and key in _NOT_FOR_HOLD:
        return "—"
    return fmt(getattr(metrics, key))


def console_summary(report: SymbolReport) -> str:
    names = [r.name for r in report.results] + ["buy_and_hold"]
    metrics: list[Metrics] = [r.metrics for r in report.results] + [report.hold.metrics]
    width = max(14, *(len(n) + 2 for n in names))
    lines = [f"\n=== {report.symbol} {report.timeframe} "
             f"{report.hold.times[0]:%Y-%m-%d} -> {report.hold.times[-1]:%Y-%m-%d} ==="]
    lines.append("".ljust(30) + "".join(n.rjust(width) for n in names))
    for key, label, fmt in METRIC_ROWS:
        cells = [_cell(key, fmt, m, i == len(metrics) - 1).rjust(width) for i, m in enumerate(metrics)]
        lines.append(label.split(" · ")[0].ljust(30) + "".join(cells))
    for result in report.results:
        randoms = report.randoms.get(result.name)
        if randoms is not None:
            lines.append(
                f"{result.name}: random-entry median return {np.median(randoms.total_returns) * 100:+.1f}%, "
                f"strategy percentile {randoms.percentile_of(result.metrics.total_return):.0f}"
            )
        for line in report.verdicts.get(result.name, []):
            lines.append(f"  - {line}")
        for note in result.notes:
            lines.append(f"  ! {note}")
    return "\n".join(lines)


def _downsample(times: list[datetime], values: np.ndarray, limit: int = 700) -> tuple[list[int], list[float]]:
    n = len(values)
    step = max(1, math.ceil(n / limit))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [int(times[i].timestamp() * 1000) for i in idx], [round(float(values[i]), 2) for i in idx]


def _drawdown(values: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(values)
    return -(peaks - values) / np.where(peaks > 0, peaks, 1)


def _chart_payload(report: SymbolReport) -> dict:
    series = []
    for k, result in enumerate(report.results + [report.hold]):
        x, y = _downsample(result.times, result.equity)
        _, dd = _downsample(result.times, _drawdown(result.equity))
        series.append({"name": result.name, "slot": k, "x": x, "y": y, "dd": [round(v * 100, 2) for v in dd]})
    band = None
    if report.results and report.results[0].name in report.randoms:
        r = report.randoms[report.results[0].name]
        times = report.hold.times[-len(r.band_mid):]
        x, low = _downsample(times, r.band_low)
        _, high = _downsample(times, r.band_high)
        band = {"x": x, "low": low, "high": high, "label": f"random entries 5–95% ({report.results[0].name})"}
    return {"series": series, "band": band}


def write_html(reports: list[SymbolReport], path: str | Path, settings_summary: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    payloads = []
    for idx, report in enumerate(reports):
        names = [r.name for r in report.results] + ["buy_and_hold"]
        metrics = [r.metrics for r in report.results] + [report.hold.metrics]
        head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
        body = "".join(
            "<tr><td>" + html.escape(label) + "</td>"
            + "".join(f"<td class=num>{html.escape(_cell(key, fmt, m, i == len(metrics) - 1))}</td>" for i, m in enumerate(metrics))
            + "</tr>"
            for key, label, fmt in METRIC_ROWS
        )
        verdict_html = ""
        for result in report.results:
            items = "".join(f"<li>{html.escape(line)}</li>" for line in report.verdicts.get(result.name, []) + result.notes)
            randoms = report.randoms.get(result.name)
            extra = ""
            if randoms is not None:
                extra = (
                    f"<div class=meta>Random entries ({randoms.runs} runs): median return "
                    f"{np.median(randoms.total_returns) * 100:+.1f}%, 95th percentile "
                    f"{np.percentile(randoms.total_returns, 95) * 100:+.1f}%</div>"
                )
            verdict_html += f"<div class=verdict><b>{html.escape(result.name)}</b>{extra}<ul>{items}</ul></div>"
        period_html = ""
        for result in report.results:
            rows = report.periods.get(result.name) or []
            if not rows:
                continue
            trs = "".join(
                f"<tr><td>{p.start:%Y-%m-%d} → {p.end:%Y-%m-%d}</td><td class=num>{_pct(p.strategy_return)}</td>"
                f"<td class=num>{p.strategy_max_dd * 100:.1f}%</td><td class=num>{p.trades}</td>"
                f"<td class=num>{_pct(p.hold_return)}</td><td class=num>{p.hold_max_dd * 100:.1f}%</td></tr>"
                for p in rows
            )
            period_html += (
                f"<h3>{html.escape(result.name)} · תקופות (פרמטרים קבועים, כל תקופה מחוץ לדגימה)</h3>"
                "<div class=scroll><table><thead><tr><th>Period</th><th>Strategy</th><th>Max DD</th><th>Trades</th>"
                f"<th>Buy &amp; hold</th><th>B&amp;H DD</th></tr></thead><tbody>{trs}</tbody></table></div>"
            )
        trade_html = ""
        for result in report.results:
            trs = "".join(
                f"<tr><td>{t.entry_time:%Y-%m-%d %H:%M}</td><td>{t.exit_time:%Y-%m-%d %H:%M}</td><td>{t.side}</td>"
                f"<td class=num>{t.entry:.6g}</td><td class=num>{t.exit:.6g}</td>"
                f"<td class='num {'pos' if t.pnl > 0 else 'neg'}'>{t.pnl:+,.2f}</td><td class=num>{t.r_multiple:+.2f}R</td>"
                f"<td>{html.escape(t.reason)}</td></tr>"
                for t in result.trades[-200:][::-1]
            )
            trade_html += (
                f"<details><summary>{html.escape(result.name)} · {len(result.trades)} trades (latest 200)</summary>"
                "<div class=scroll><table><thead><tr><th>Entry</th><th>Exit</th><th>Side</th><th>Entry px</th>"
                f"<th>Exit px</th><th>PnL</th><th>R</th><th>Reason</th></tr></thead><tbody>{trs}</tbody></table></div></details>"
            )
        sections.append(
            f"""
<section class=card>
  <h2>{html.escape(report.symbol)} · {html.escape(report.timeframe)} ·
      {report.hold.times[0]:%Y-%m-%d} → {report.hold.times[-1]:%Y-%m-%d}</h2>
  <div class=verdicts>{verdict_html}</div>
  <h3>Equity · עקומת הון</h3>
  <div class=chart id=eq{idx}></div>
  <div class=legend id=lg{idx}></div>
  <h3>Drawdown · ירידה מהשיא</h3>
  <div class=chart id=dd{idx}></div>
  <h3>Metrics · מדדים</h3>
  <div class=scroll><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>
  {period_html}
  {trade_html}
</section>"""
        )
        payloads.append(_chart_payload(report))
    settings_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td class=num>{html.escape(str(v))}</td></tr>" for k, v in settings_summary.items()
    )
    document = _TEMPLATE.replace("__SECTIONS__", "\n".join(sections))
    document = document.replace("__SETTINGS__", settings_rows)
    document = document.replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M"))
    document = document.replace("__DATA__", json.dumps(payloads))
    document = document.replace("__COLORS__", json.dumps(SERIES))
    path.write_text(document, encoding="utf-8")
    return path


_TEMPLATE = """<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Report</title>
<style>
:root { color-scheme: light; --bg:#f4f3f0; --surface:#fcfcfb; --border:#e2e0da; --text:#0b0b0b; --text-2:#52514e;
  --muted:#8a8984; --grid:#e8e6e1; --band:#d9d7d1; --good:#0a7d0a; --bad:#c42f2f; --dark:0; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --bg:#121211; --surface:#1a1a19; --border:#2f2f2c;
  --text:#fff; --text-2:#c3c2b7; --muted:#8f8e86; --grid:#2a2a28; --band:#33332f; --good:#3fbf3f; --bad:#ef6b6b; --dark:1; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,-apple-system,"Segoe UI",Arial,sans-serif; }
main { max-width:1200px; margin:0 auto; padding:16px; display:grid; gap:16px; }
h1 { font-size:20px; margin:8px 0 0; } h2 { font-size:16px; margin:0 0 10px; } h3 { font-size:13px; color:var(--text-2); margin:18px 0 8px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; min-width:0; }
.warn { background:#fdf0d2; color:#7a4f00; border-radius:12px; padding:12px 14px; }
@media (prefers-color-scheme: dark) { .warn { background:#362a10; color:#f0b43a; } }
.num { font-variant-numeric:tabular-nums; direction:ltr; unicode-bidi:isolate; text-align:end; }
.pos { color:var(--good); } .neg { color:var(--bad); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:6px 8px; border-bottom:1px solid var(--grid); text-align:start; white-space:nowrap; }
th { color:var(--text-2); font-weight:600; }
th:not(:first-child) { text-align:end; }
.scroll { overflow-x:auto; }
.chart { position:relative; height:260px; }
.chart svg { width:100%; height:100%; display:block; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:12.5px; color:var(--text-2); margin-top:6px; }
.legend i { display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-inline-end:6px; }
.legend i.band { height:10px; background:var(--band); }
.tip { position:absolute; pointer-events:none; background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:6px 9px; font-size:12px; display:none; box-shadow:0 4px 16px rgba(0,0,0,.12); white-space:nowrap; direction:ltr; }
.verdicts { display:grid; gap:10px; }
.verdict { border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
.verdict ul { margin:6px 0 0; padding-inline-start:18px; }
.meta { color:var(--text-2); font-size:12.5px; }
details { margin-top:12px; } summary { cursor:pointer; color:var(--text-2); }
</style>
</head>
<body>
<main>
  <h1>Backtest report · דוח בקטסט</h1>
  <div class=warn><b>Not proof of future profit.</b> Past data, estimated fees and slippage, no headline sentiment. If the strategy cannot beat random entries with the same risk rules, it has no demonstrated edge.<div dir=rtl lang=he style="margin-top:6px"><b>זה לא הוכחה לרווח עתידי.</b> נתוני עבר, עמלות והחלקה מוערכות, ללא סנטימנט כותרות. אם האסטרטגיה לא מנצחת
  כניסות אקראיות עם אותם כללי סיכון, אין לה יתרון מוכח.</div></div>
  __SECTIONS__
  <section class=card><details><summary>Settings used · הגדרות</summary><table><tbody>__SETTINGS__</tbody></table></details>
  <div class=meta>Generated __GENERATED__</div></section>
</main>
<script>
const DATA = __DATA__;
const COLORS = __COLORS__;
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const color = (slot) => COLORS[slot % COLORS.length][dark ? 1 : 0];
const fmtDate = (ms) => new Date(ms).toISOString().slice(0, 10);
function chart(el, series, key, band, unit) {
  const W = el.clientWidth || 800, H = 260, L = 8, R = 70, T = 10, B = 24;
  const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s[key]);
  if (band) ys.push(...band.low, ...band.high);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), lo = Math.min(...ys), hi = Math.max(...ys);
  if (key === "dd") hi = 0;
  const pad = (hi - lo) * 0.06 || 1; lo -= pad; if (key !== "dd") hi += pad;
  const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - R), Y = v => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4, y = Y(v);
    g += `<line x1=${L} x2=${W - R} y1=${y} y2=${y} stroke="var(--grid)"/><text x=${W - R + 6} y=${y + 4} font-size=11 fill="var(--muted)">${unit === "%" ? v.toFixed(0) + "%" : Math.round(v).toLocaleString()}</text>`; }
  let b = "";
  if (band && key === "y") {
    const up = band.x.map((x, i) => `${X(x)},${Y(band.high[i])}`).join(" ");
    const dn = band.x.map((x, i) => `${X(x)},${Y(band.low[i])}`).reverse().join(" ");
    b = `<polygon points="${up} ${dn}" fill="var(--band)" opacity=".8"/>`;
  }
  const lines = series.map(s => `<polyline fill=none stroke="${color(s.slot)}" stroke-width=2 stroke-linejoin=round points="${s.x.map((x, i) => `${X(x)},${Y(s[key][i])}`).join(" ")}"/>`).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${g}${b}${lines}
    <text x=${L} y=${H - 6} font-size=11 fill="var(--muted)">${fmtDate(x0)}</text>
    <text x=${W - R} y=${H - 6} font-size=11 fill="var(--muted)" text-anchor=end>${fmtDate(x1)}</text>
    <line class=cx y1=${T} y2=${H - B} stroke="var(--text-2)" stroke-dasharray="3 3" visibility=hidden />
    <rect x=${L} y=0 width=${W - L - R} height=${H} fill=transparent /></svg><div class=tip></div>`;
  const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), cx = el.querySelector(".cx");
  svg.querySelector("rect").addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect(), mx = (ev.clientX - r.left) * W / r.width;
    const tv = x0 + (mx - L) / (W - L - R) * (x1 - x0);
    const s0 = series[0]; let i = 0, best = Infinity;
    s0.x.forEach((x, k) => { const d = Math.abs(x - tv); if (d < best) { best = d; i = k; } });
    const px = X(s0.x[i]); cx.setAttribute("x1", px); cx.setAttribute("x2", px); cx.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>${fmtDate(s0.x[i])}</b><br>` + series.map(s => {
      let j = 0, bd = Infinity; s.x.forEach((x, k) => { const d = Math.abs(x - s0.x[i]); if (d < bd) { bd = d; j = k; } });
      const v = s[key][j]; return `<span style="color:${color(s.slot)}">■</span> ${s.name}: ${unit === "%" ? v.toFixed(1) + "%" : Math.round(v).toLocaleString()}`;
    }).join("<br>");
    tip.style.display = "block";
    const left = px * r.width / W; tip.style.left = Math.min(Math.max(0, left + 12), r.width - tip.offsetWidth) + "px"; tip.style.top = "8px";
  });
  svg.querySelector("rect").addEventListener("mouseleave", () => { tip.style.display = "none"; cx.setAttribute("visibility", "hidden"); });
}
function draw() {
  DATA.forEach((d, i) => {
    chart(document.getElementById("eq" + i), d.series, "y", d.band, "");
    chart(document.getElementById("dd" + i), d.series, "dd", null, "%");
    document.getElementById("lg" + i).innerHTML = d.series.map(s => `<span><i style="background:${color(s.slot)}"></i>${s.name}</span>`).join("")
      + (d.band ? `<span><i class=band></i>${d.band.label}</span>` : "");
  });
}
draw(); addEventListener("resize", draw);
</script>
</body>
</html>
"""
