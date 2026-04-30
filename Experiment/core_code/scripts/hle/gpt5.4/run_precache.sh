#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/gpt5.4_no_integrate
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/hle/gpt5.4/hle_gpt5.4_precache.yaml \
	> ../analysis/run/hle/gpt5.4_no_integrate/precache.log 2>&1 &
