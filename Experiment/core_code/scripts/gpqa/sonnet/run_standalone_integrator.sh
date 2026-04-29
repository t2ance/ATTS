#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/gpqa/sonnet_standalone_integrator
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/gpqa/sonnet/gpqa_sonnet_standalone_integrator.yaml \
	> ../analysis/run/gpqa/sonnet_standalone_integrator/run.log 2>&1 &
