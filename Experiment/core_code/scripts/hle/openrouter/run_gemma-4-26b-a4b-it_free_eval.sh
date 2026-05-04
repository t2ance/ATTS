#!/usr/bin/env bash
set -euo pipefail

# HLE eval (ATTS over precached explores) for google/gemma-4-26b-a4b-it:free.
# Companion: ../../scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_eval.yaml
# Companion todo: ../../todo_openrouter_hle_gemma-4-26b-a4b-it_free.md item 04.
# Prerequisite:
#   1. Qwen3.6 vLLM serve up on localhost:8000.
#   2. Precache complete: 400/400 result.json under
#      ../analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free/gold/

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp ../analysis/run/hle/openrouter_google_gemma-4-26b-a4b-it_free

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_gemma-4-26b-a4b-it_free_eval_${DATE}.log"

eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
  --config scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_eval.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
