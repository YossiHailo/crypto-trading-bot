"""The stdlib daily report reads the bot's database correctly."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from core.models import CloseReason, RiskAction, RiskDecision, SignalAction
from core.state import PortfolioStore
from scripts.daily_report import build, open_copy, to_markdown
from tests.conftest import technical

DAY = date(2026, 10, 8)
# 00:00 Israel time (UTC+3) = 21:00 UTC the day before.
START = datetime(2026, 10, 7, 21, 0, tzinfo=timezone.utc)


def test_daily_report_counts_uptime_trades_and_reasons(tmp_path) -> None:
    clock = {"now": START}
    store = PortfolioStore(tmp_path / "p.sqlite", 10_000, fee_rate=0.0, clock=lambda: clock["now"])
    signal = technical(action=SignalAction.HOLD, confidence=0.0, price=100.0)
    hold = RiskDecision(RiskAction.NO_ACTION, "BTC/USDT", 0, 100, 0, 0, 0, "Technical signal is HOLD. x")
    # 8 hours of checks, then a 3-hour gap (computer asleep), then 2 more hours.
    for minute in list(range(0, 8 * 60, 15)) + list(range(11 * 60, 13 * 60, 15)):
        clock["now"] = START + timedelta(minutes=minute)
        store.record_decision(signal, hold, 0.2)
        store.record_equity({"BTC/USDT": 100.0})
    clock["now"] = START + timedelta(hours=2)
    store.open_position(symbol="BTC/USDT", side="long", size=10, price=100, stop_loss=95, take_profit=110)
    clock["now"] = START + timedelta(hours=5)
    store.close_position("BTC/USDT", 110, CloseReason.TAKE_PROFIT)

    report = build(open_copy(tmp_path / "p.sqlite"), DAY, "Asia/Jerusalem", 15, START + timedelta(hours=13))
    assert report["uptime"]["checks_ran"] == 40
    assert report["uptime"]["checks_expected"] == 52
    assert any(g["minutes"] >= 180 for g in report["uptime"]["gaps"])
    assert report["trades"]["closed"] == 1 and report["trades"]["wins"] == 1
    assert report["trades"]["pnl"] == pytest.approx(100)
    assert report["decisions"]["by_reason"] == {"אין אות": 40}
    assert report["coins"]["BTC/USDT"]["checks"] == 40
    assert "דוח יומי 2026-10-08" in to_markdown(report)
    store.close()


def test_daily_report_separates_deposits_from_trading_results(tmp_path) -> None:
    # Account opened Oct 31 (Israel), the Nov 1 00:00 UTC deposit lands on the Nov 1 report.
    opened = datetime(2026, 10, 31, 10, tzinfo=timezone.utc)
    clock = {"now": opened}
    store = PortfolioStore(tmp_path / "d.sqlite", 1_000, fee_rate=0.0, clock=lambda: clock["now"])
    store.record_equity({})
    day_start = datetime(2026, 10, 31, 22, tzinfo=timezone.utc)  # 00:00 Israel (UTC+2 in November)
    for minute in range(0, 24 * 60, 15):
        clock["now"] = day_start + timedelta(minutes=minute)
        store.apply_monthly_deposits(100, 1)
        store.record_equity({})
    report = build(open_copy(tmp_path / "d.sqlite"), date(2026, 11, 1), "Asia/Jerusalem", 15, day_start + timedelta(days=1))
    account = report["account"]
    assert account["deposited_today"] == pytest.approx(100)
    assert account["change"] == pytest.approx(0)  # value rose by 100, all of it the deposit
    assert account["deposited_total"] == pytest.approx(1_100)
    assert account["since_start"] == pytest.approx(0)
    assert "הפקדה חודשית" in to_markdown(report)
    store.close()


def test_daily_report_html_page_and_index(tmp_path) -> None:
    from scripts.daily_report import main as report_main

    clock = {"now": START}
    store = PortfolioStore(tmp_path / "h.sqlite", 1_000, fee_rate=0.0, clock=lambda: clock["now"])
    store.record_equity({})
    store.close()
    out = tmp_path / "out"
    assert report_main(["--db", str(tmp_path / "h.sqlite"), "--out", str(out), "--date", "2026-10-08"]) == 0
    page = (out / "2026-10-08.html").read_text(encoding="utf-8")
    assert '<meta charset="utf-8">' in page and "<h1>דוח יומי 2026-10-08</h1>" in page and "<table>" in page
    assert '2026-10-08.html' in (out / "index.html").read_text(encoding="utf-8")


def test_daily_report_includes_the_strategy_tournament(tmp_path) -> None:
    import json

    from scripts.daily_report import main as report_main

    clock = {"now": START}
    store = PortfolioStore(tmp_path / "a.sqlite", 1_000, fee_rate=0.0, clock=lambda: clock["now"])
    store.record_equity({})
    store.close()
    (tmp_path / "arena").mkdir()
    (tmp_path / "arena" / "arena.json").write_text(json.dumps({
        "champion": "rsi_dip",
        "standings": [
            {"strategy": "rsi_dip", "label_he": "קנייה בירידה זמנית", "status": "active", "equity": 1010, "deposits": 1000,
             "true_profit": 10, "true_profit_pct": 0.01, "trades": 2, "wins": 1, "champion": True},
            {"strategy": "trend", "label_he": "מעקב מגמה", "status": "eliminated", "equity": 890, "deposits": 1000,
             "true_profit": -110, "true_profit_pct": -0.11, "trades": 5, "wins": 1, "champion": False},
        ],
        "history": [{"at": "2026-10-08T05:00:00+00:00", "from": "trend", "to": "rsi_dip", "reason": "trend was eliminated"}],
    }), encoding="utf-8")
    out = tmp_path / "out"
    assert report_main(["--db", str(tmp_path / "a.sqlite"), "--out", str(out), "--date", "2026-10-08"]) == 0
    data = json.loads((out / "2026-10-08.json").read_text(encoding="utf-8"))
    assert data["arena"]["champion"] == "rsi_dip"
    assert data["arena"]["eliminated"] == ["trend"]
    assert len(data["arena"]["switches_today"]) == 1
    md = (out / "2026-10-08.md").read_text(encoding="utf-8")
    assert "תחרות השיטות" in md and "הודחה" in md
