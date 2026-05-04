#!/usr/bin/env bash
set -euo pipefail

# HLE eval (n=100 ATTS) for deepseek/deepseek-v4-pro via OpenRouter (paid).
# Companion: scripts/hle/openrouter/hle_deepseek-v4-pro_eval.yaml
# Companion todo: todo_openrouter_hle_deepseek-v4-pro.md item 04.
# Reads from precache; only Haiku judge calls cost money on first pass.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_deepseek-v4-pro_eval_${DATE}.log"

# OPENROUTER_API_KEY: pulled from active export line in ~/.bashrc (line 147 as
#   of 2026-05-04). See companion smoke .sh for rationale on this pattern.
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
  --config scripts/hle/openrouter/hle_deepseek-v4-pro_eval.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
