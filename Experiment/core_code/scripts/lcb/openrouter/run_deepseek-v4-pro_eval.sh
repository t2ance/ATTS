#!/usr/bin/env bash
set -euo pipefail

# LCB eval (n=100 ATTS) for deepseek/deepseek-v4-pro via OpenRouter (paid).
# Companion: scripts/lcb/openrouter/lcb_deepseek-v4-pro_eval.yaml
# Companion todo: todo_openrouter_lcb_deepseek-v4-pro.md item 04.
# Reads from precache; LCB grading is local code execution (free).

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_lcb_deepseek-v4-pro_eval_${DATE}.log"

# OPENROUTER_API_KEY: pulled from active export line in ~/.bashrc (line 147 as
#   of 2026-05-04). See companion smoke .sh for rationale.
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
  --config scripts/lcb/openrouter/lcb_deepseek-v4-pro_eval.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
