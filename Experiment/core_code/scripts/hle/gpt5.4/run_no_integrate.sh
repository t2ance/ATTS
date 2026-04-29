#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/gpt5.4_no_integrate
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/hle/gpt5.4/hle_gpt5.4_no_integrate.yaml \
	> ../analysis/run/hle/gpt5.4_no_integrate/delegated.log 2>&1 &
