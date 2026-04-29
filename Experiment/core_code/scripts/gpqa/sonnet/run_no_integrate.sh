#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/gpqa/sonnet/gpqa_sonnet_no_integrate.yaml \
	> ../analysis/run/gpqa/sonnet_no_integrate/delegated.log 2>&1 &
