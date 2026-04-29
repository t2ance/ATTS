#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/gpqa/gpt5.2_no_integrate
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--config configs/gpqa_gpt5.2_precache.yaml \
	> ../analysis/run/gpqa/gpt5.2_no_integrate/precache.log 2>&1 &
