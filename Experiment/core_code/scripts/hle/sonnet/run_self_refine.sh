#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/hle/sonnet/hle_sonnet_self_refine.yaml \
	> ../analysis/run/hle/sonnet_self_refine/self_refine.log 2>&1 &
