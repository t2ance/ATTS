#!/usr/bin/env bash
set -euo pipefail

# Generate ATTS trajectories for HLE training data.
# Uses Sonnet orchestrator + cached Haiku explores.
# Runs on all 668 Gold text-only questions (first 100 = eval, rest = training).

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

unset CLAUDECODE 2>/dev/null || true

PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
    --backend claude \
    --method tts-agent \
    --subset gold \
    --num 668 \
    --seed 42 \
    --num-explores 8 \
    --num-workers 8 \
    --text-only \
    --log-dir ../analysis/run/hle/sonnet_training \
    --orchestrator-model claude-sonnet-4-6 \
    --explore-model claude-haiku-4-5-20251001 \
    --cache-dirs ../analysis/cache/hle/haiku/gold \
    --cache-only \
    --max-output-chars 3500 \
    > ./tmp/generate_hle_trajectories.log 2>&1 &

echo "Started trajectory generation. PID: $!"
echo "Log: ./tmp/generate_hle_trajectories.log"
