# Changelog

## 1.1.0

### Safety and correctness (live)

- Decisions use **closed candles only**. The forming candle is dropped, so a signal no longer flips inside a bar when the loop runs every 15 minutes.
- **Idempotent order placement** with `clientOrderId`. A timeout no longer re-sends an entry that the exchange already accepted.
- The attached `stopLoss`/`takeProfit` bracket path was removed (some venues ignore those parameters silently, which left positions marked as protected with no stop). Entries are now followed by a **separate, tracked stop order** whose id is stored.
- Stops and targets are placed at the planned distance from the **actual fill price**, not the candle close.
- **Every close cancels the stored protective orders first.** Previously a spot stop locked the coins and the signal close failed forever; on swap orphan stops could hit a later position.
- Spot sells use the amount actually held (exchanges often take the buy fee in the coin).
- On spot, the take-profit is managed by the bot (two sell orders would lock the same coins). On swap, a reduce-only TP is placed.
- When the exchange stop fires, **PnL uses that order's fill price**, not the current market price.
- Backstop: if the price stays beyond a protected stop for two checks, the bot closes at market.
- **Startup reconciliation**: positions closed while offline are recorded, missing stops are re-placed, orphan bot orders are cancelled.
- Resting limit entries expire (`PENDING_ORDER_TTL_MINUTES`); partial fills are adopted and protected. Limit fills now get exchange protection too.
- **Swap support fixed**: leverage and isolated margin are set at startup (`LEVERAGE`), and accounting locks margin instead of the full notional, so equity and the daily breaker are correct on swap.

### Risk and finance

- Default per-position cap `MAX_NOTIONAL_PCT=0.5` and new `MAX_TOTAL_EXPOSURE_PCT=1.0`. Previously almost every trade was all-in because tight stops asked for 2–3x equity and only the cash cap stopped it.
- The decision text reports the **effective risk** as a percentage of equity.
- **Reward:risk is checked after fees and slippage.** The old check always equalled `REWARD_RISK_RATIO`, so `MIN_REWARD_RISK` never rejected anything.
- **Max-drawdown kill switch** (`MAX_DRAWDOWN_PCT`, default 20% from the equity peak). It stays on until reset from the dashboard.
- Hard limits on settings: `RISK_PER_TRADE` ≤ 5%, leverage only on swap, minimum candle window per strategy, token required for a non-local dashboard.
- Paper mode now charges `SLIPPAGE_PCT` on market fills and detects stops touched by a wick between checks (stop assumed first when a bar touches both).

### Strategy and sentiment

- Strategy interface with `trend` (new default: Donchian breakout filtered by EMA50/EMA200, exit on a 10-bar channel) and `legacy` (the original mix, kept for comparison). Defaults moved to `TIMEFRAME=4h`, `ATR_MULTIPLIER=2.5`, `REWARD_RISK_RATIO=3`, `RISK_PER_TRADE=0.01`. **Neither strategy has a demonstrated edge; run the backtest.**
- Headline scoring counts per distinct headline and handles negation ("SEC delays spot ETF approval" is no longer bullish).

### New tools

- **Historical backtester** (`python main.py backtest`): CCXT download with CSV cache, next-bar-open fills, buy-and-hold and random-entry benchmarks, period breakdown, optional Fear & Greed history, HTML report.
- **Local web dashboard** (http://127.0.0.1:8765): account, positions, signals, every decision with its reason, trades, warnings, equity curve, and controls (pause, resume, run now, reset kill switch, close all, stop). Hebrew and English, light and dark. Standard library only.
- `scripts/demo_dashboard.py` previews the dashboard with fake data.
- Every decision and an equity snapshot per cycle are stored in SQLite (old databases are migrated automatically).

### Operations

- Secrets are masked everywhere: logs (including tracebacks), Telegram messages and the dashboard. The CryptoPanic token used to appear in error messages.
- Market data for all symbols is fetched concurrently.
- Docker: health check from the heartbeat, dashboard published on 127.0.0.1 only, `reports/` volume.
- Clearer configuration errors at startup.
