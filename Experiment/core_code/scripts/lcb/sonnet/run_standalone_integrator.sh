#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
export PYTHONPATH="../code_references/LiveCodeBench:${PYTHONPATH:-}"
mkdir -p ../analysis/run/lcb/sonnet_standalone_integrator
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/lcb/sonnet/lcb_sonnet_standalone_integrator.yaml \
	> ../analysis/run/lcb/sonnet_standalone_integrator/run.log 2>&1 &
