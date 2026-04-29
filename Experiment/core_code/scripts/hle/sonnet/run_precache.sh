#!/usr/bin/env bash
set -euo pipefail

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--config scripts/hle/sonnet/hle_sonnet_precache.yaml \
	> ../analysis/run/hle/sonnet/precache.log 2>&1 &
