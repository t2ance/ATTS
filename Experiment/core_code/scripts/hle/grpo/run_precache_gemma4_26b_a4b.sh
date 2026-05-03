#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/gemma4_26b_a4b_it/
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/hle/grpo/hle_gemma4_26b_a4b_precache.yaml \
	> ../analysis/run/hle/gemma4_26b_a4b_it/precache.log 2>&1 &
echo "HLE precache PID $!"
echo "log: ../analysis/run/hle/gemma4_26b_a4b_it/precache.log"
