#!/usr/bin/env bash
set -euo pipefail

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/gpt5.4_high
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
    --config scripts/hle/gpt5.4_high/hle_gpt5.4_high_precache.yaml \
    > ../analysis/run/hle/gpt5.4_high/precache.log 2>&1 &
echo "HLE 5.4-high precache PID=$!"
