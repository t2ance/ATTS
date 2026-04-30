#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/babyvision/sonnet/babyvision_sonnet_precache.yaml \
	> ../analysis/run/babyvision/sonnet/precache.log 2>&1 &
