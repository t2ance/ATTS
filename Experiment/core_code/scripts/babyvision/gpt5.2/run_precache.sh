#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/babyvision/gpt5.2_no_integrate_high
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/babyvision/gpt5.2/babyvision_gpt5.2_precache.yaml \
	> ../analysis/run/babyvision/gpt5.2_no_integrate_high/precache.log 2>&1 &
