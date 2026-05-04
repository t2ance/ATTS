#!/usr/bin/env bash
set -euo pipefail

# HLE production precache (400 explores) for google/gemma-4-26b-a4b-it:free.
# Companion: ../../scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_precache.yaml
# Companion todo: ../../todo_openrouter_hle_gemma-4-26b-a4b-it_free.md item 03.
# Prerequisite: Qwen3.6 vLLM serve must be up on localhost:8000.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

DATE=$(date +%Y%m%d_%H%M%S)
LOG="/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_gemma-4-26b-a4b-it_free_precache_${DATE}.log"

eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
  --config scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_precache.yaml \
  > "$LOG" 2>&1 &

echo "started PID $!"
echo "log: $LOG"
