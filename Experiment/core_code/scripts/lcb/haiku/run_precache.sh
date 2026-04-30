#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/lcb/haiku/lcb_haiku_precache.yaml \
	> ../analysis/run/lcb/haiku/precache.log 2>&1 &
