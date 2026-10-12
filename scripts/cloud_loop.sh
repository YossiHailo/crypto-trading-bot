#!/usr/bin/env bash
# Runs one check every 15 minutes (on the quarter hour) until LOOP_MINUTES have passed,
# saving the state after every check. The workflow then starts its own next run, so the
# bot keeps going even when GitHub's scheduler is late or skips scheduled runs.
set -uo pipefail
LOOP_MINUTES="${LOOP_MINUTES:-330}"
STEP=$(( ${LOOP_INTERVAL_MINUTES:-15} * 60 ))
deadline=$(( $(date +%s) + LOOP_MINUTES * 60 ))
action="${ACTION:-none}"
# Historical strategy tournament (about 10 minutes): on request, or once if it never ran.
# A failed attempt leaves a marker so later runs don't retry it on their own every time.
if [ "${ARENA_ENABLED:-false}" = "true" ] && { [ "$action" = "research" ] || { [ ! -f state/data/arena/research.json ] && [ ! -f state/data/arena/research.failed ]; }; }; then
  echo "::group::Research: every strategy on 3 years of history"
  mkdir -p state/data/arena
  if timeout 45m python scripts/research.py --html state/docs/research/index.html; then
    rm -f state/data/arena/research.failed
  else
    date -u +%FT%TZ > state/data/arena/research.failed
    echo "::warning::Research failed; the tournament continues without it. Run the workflow with action=research to retry."
  fi
  echo "::endgroup::"
fi
[ "$action" = "research" ] && action=none
while :; do
  echo "::group::Check at $(date -u +%FT%TZ) (action: $action)"
  python main.py --once --action "$action" --export-state state/docs/state.json || echo "::warning::The check failed; trying again at the next slot."
  bash scripts/cloud_save.sh || echo "::warning::Saving the state failed; trying again after the next check."
  echo "::endgroup::"
  action=none  # a remote-control action runs only once
  now=$(date +%s)
  next=$(( (now / STEP + 1) * STEP ))
  if [ "$next" -ge "$deadline" ]; then
    echo "Loop finished; the next run takes over."
    break
  fi
  sleep $(( next - now ))
done
