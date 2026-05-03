#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/babyvision/gemma4_26b_a4b_it/
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/babyvision/grpo/babyvision_gemma4_26b_a4b_precache.yaml \
	> ../analysis/run/babyvision/gemma4_26b_a4b_it/precache.log 2>&1 &
echo "BV precache PID $!"
echo "log: ../analysis/run/babyvision/gemma4_26b_a4b_it/precache.log"
