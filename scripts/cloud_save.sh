#!/usr/bin/env bash
# Save the bot's state (database, dashboard snapshot, reports, logs) to the `state` branch.
# Used by .github/workflows/bot.yml after every check.
set -euo pipefail
TZNAME="${REPORT_TZ:-Asia/Jerusalem}"
INTERVAL="${LOOP_INTERVAL_MINUTES:-15}"
# Fold each database's write-ahead log into the database file before saving it.
for db in state/data/portfolio.sqlite state/data/arena/*.sqlite; do
  [ -f "$db" ] && python -c "import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()" "$db"
done
if [ -f state/data/portfolio.sqlite ]; then
  today=$(TZ="$TZNAME" date +%F)
  yesterday=$(TZ="$TZNAME" date -d yesterday +%F)
  python scripts/daily_report.py --db state/data/portfolio.sqlite --out state/reports/daily --tz "$TZNAME" --interval "$INTERVAL" --date "$yesterday" > /dev/null || true
  python scripts/daily_report.py --db state/data/portfolio.sqlite --out state/reports/daily --tz "$TZNAME" --interval "$INTERVAL" --date "$today" > state/reports/today.md || true
fi
mkdir -p state/docs
cp dashboard/static/index.html state/docs/index.html
if [ -d state/reports/daily ]; then mkdir -p state/docs/reports && cp state/reports/daily/* state/docs/reports/; fi
touch state/docs/.nojekyll
# Keep logs small: the last 3000 lines are plenty for troubleshooting.
for f in state/logs/*.log; do [ -f "$f" ] && tail -n 3000 "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
rm -f state/logs/*.log.[0-9]*
cd state
rm -rf .git
git init --quiet
git config user.name "trading-bot"
git config user.email "trading-bot@users.noreply.github.com"
git checkout --quiet --orphan state
git add -A
git commit --quiet -m "State after run ${GITHUB_RUN_NUMBER:-local} ($(date -u +%FT%TZ))"
git push --quiet --force "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" state
rm -rf .git
echo "State saved."
