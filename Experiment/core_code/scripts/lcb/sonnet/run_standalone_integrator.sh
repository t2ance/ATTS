#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
export PYTHONPATH="../code_references/LiveCodeBench:${PYTHONPATH:-}"
mkdir -p ../analysis/run/lcb/sonnet_standalone_integrator
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark lcb \
	--backend claude \
	--method standalone-integrator \
	--seed 42 \
	--num-explores 8 \
	--num-workers 16 \
	--log-dir ../analysis/run/lcb/sonnet_standalone_integrator \
	--explore-model claude-sonnet-4-6 \
	--orchestrator-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/lcb/sonnet \
	> ../analysis/run/lcb/sonnet_standalone_integrator/run.log 2>&1 &
