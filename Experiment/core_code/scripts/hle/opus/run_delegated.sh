#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/hle_opus_delegated.yaml \
	> ../analysis/run/hle/opus/gold_delegated.log 2>&1 &
