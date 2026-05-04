#!/usr/bin/env bash
set -euo pipefail

# LCB pre-flight smoke for deepseek/deepseek-v4-pro via OpenRouter (paid).
# Companion: scripts/lcb/openrouter/lcb_deepseek-v4-pro_smoke.yaml
# Companion todo: todo_openrouter_lcb_deepseek-v4-pro.md item 02.
# 4 explores total (num=2 × num_explores=2). Goal: validate pipeline + lcb_runner.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_lcb_deepseek-v4-pro_smoke_${DATE}.log"

# OPENROUTER_API_KEY: pulled from active export line in ~/.bashrc (line 147 as
#   of 2026-05-04). Bypasses .bashrc non-interactive guard.
#   See /home/peijia/dr-claw/Explain/CLAUDE.md "API-key freshness in
#   long-running shells" for rationale.
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
  --config scripts/lcb/openrouter/lcb_deepseek-v4-pro_smoke.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
