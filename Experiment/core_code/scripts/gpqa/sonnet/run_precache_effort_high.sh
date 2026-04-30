#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/gpqa/sonnet/gpqa_sonnet_precache_effort_high.yaml \
	> ../analysis/run/gpqa/sonnet_effort_high_precache4.log 2>&1 &
