#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/sonnet/hle_sonnet_delegated.yaml \
	> ../analysis/run/hle/sonnet/gold_delegated.log 2>&1 &
