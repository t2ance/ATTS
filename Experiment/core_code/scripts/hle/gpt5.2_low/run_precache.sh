#!/usr/bin/env bash
set -euo pipefail

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--config configs/hle_gpt5.2_low_precache.yaml \
	> ../analysis/run/hle/gpt5.2_low/precache.log 2>&1 &
