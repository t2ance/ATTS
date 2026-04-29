#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup env -u CLAUDECODE python precache_explores.py \
	--config configs/lcb_opus_precache.yaml \
	> ../analysis/cache/lcb/opus_precache.log 2>&1 &
