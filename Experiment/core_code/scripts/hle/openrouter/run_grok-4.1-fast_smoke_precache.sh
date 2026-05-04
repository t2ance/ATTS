#!/usr/bin/env bash
set -euo pipefail

# HLE 10-question cost-estimation precache for x-ai/grok-4.1-fast.
# Companion: scripts/hle/openrouter/hle_grok-4.1-fast_smoke_precache.yaml
# 40 explores total (num=10 × num_explores=4). Goal: estimate full HLE-100 cost.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_grok-4.1-fast_smoke_precache_${DATE}.log"

# OPENROUTER_API_KEY: pulled from active export line in ~/.bashrc.
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
  --config scripts/hle/openrouter/hle_grok-4.1-fast_smoke_precache.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
