#!/usr/bin/env bash
set -euo pipefail

# HLE eval-side smoke (num=2) for x-ai/grok-4.1-fast.
# Companion: scripts/hle/openrouter/hle_grok-4.1-fast_smoke.yaml

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_grok-4.1-fast_smoke_${DATE}.log"

eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
  --config scripts/hle/openrouter/hle_grok-4.1-fast_smoke.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
