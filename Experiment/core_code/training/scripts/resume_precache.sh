#!/usr/bin/env bash
set -euo pipefail

# Resume pre-caching after rate limit reset.
# The precache script auto-skips completed explores.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

echo "Waiting until 7:05 UTC for rate limit reset..."
while [ "$(date -u +%H%M)" -lt "0705" ]; do
    sleep 60
done

echo "Rate limit should be reset. Starting pre-cache at $(date -u)..."

PYTHONUNBUFFERED=1 python precache_explores.py \
    --backend claude \
    --cache-dirs ../analysis/cache/hle/haiku/gold \
    --subset gold \
    --num 668 \
    --num-explores 8 \
    --num-workers 4 \
    --seed 42 \
    --text-only \
    --explore-model claude-haiku-4-5-20251001 \
    --effort low \
    >> ./tmp/precache_hle_training.log 2>&1

echo "Pre-cache complete at $(date -u)"
