#!/usr/bin/env bash
set -euo pipefail

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/hle/opus/hle_opus_precache.yaml \
	> ../analysis/run/hle/opus/precache.log 2>&1 &
