#!/usr/bin/env bash
set -euo pipefail

# Pre-cache Haiku explores for ALL 668 Gold text-only HLE questions.
# The first 100 overlap with eval set (wasted but cheap).
# Training data builder will exclude them.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

PYTHONUNBUFFERED=1 nohup python precache_explores.py \
    --backend claude \
    --cache-dirs ../analysis/cache/hle/haiku/gold \
    --subset gold \
    --num 668 \
    --num-explores 8 \
    --num-workers 8 \
    --seed 42 \
    --text-only \
    --explore-model claude-haiku-4-5-20251001 \
    --effort low \
    > ./tmp/precache_hle_training.log 2>&1 &

echo "Started pre-caching. PID: $!"
echo "Log: ./tmp/precache_hle_training.log"
