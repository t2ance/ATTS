#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/gpt5.2_low/hle_gpt5.2_low_delegated.yaml \
	> ../analysis/run/hle/gpt5.2_low/gold_delegated.log 2>&1 &
