#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark lcb \
	--backend claude \
	--seed 42 \
	--num-explores 8 \
	--num-workers 2 \
	--log-dir ../analysis/run/lcb/sonnet_no_integrate \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/lcb/sonnet \
	--no-integrate \
	> ../analysis/run/lcb/sonnet_no_integrate/delegated.log 2>&1 &
