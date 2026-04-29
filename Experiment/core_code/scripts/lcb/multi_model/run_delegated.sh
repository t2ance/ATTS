#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/lcb/multi_model/lcb_multi_delegated.yaml \
	> ../analysis/run/lcb/multi_model/delegated.log 2>&1 &
