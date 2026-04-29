#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--config configs/lcb_sonnet_precache.yaml \
	> ../analysis/run/lcb/sonnet/precache.log 2>&1 &
