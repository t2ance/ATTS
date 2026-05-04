#!/usr/bin/env bash
set -euo pipefail

# LCB production precache (800 explores) for deepseek/deepseek-v4-pro via OpenRouter (paid).
# Companion: scripts/lcb/openrouter/lcb_deepseek-v4-pro_precache.yaml
# Companion todo: todo_openrouter_lcb_deepseek-v4-pro.md item 03.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_lcb_deepseek-v4-pro_precache_${DATE}.log"

# OPENROUTER_API_KEY: pulled from active export line in ~/.bashrc (line 147 as
#   of 2026-05-04). See companion smoke .sh for rationale.
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
  --config scripts/lcb/openrouter/lcb_deepseek-v4-pro_precache.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
