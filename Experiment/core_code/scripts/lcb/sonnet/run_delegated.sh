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
	--num-workers 16 \
	--log-dir ../analysis/run/lcb/sonnet \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/lcb/sonnet \
	--resume ../analysis/run/lcb/sonnet/run_20260308_230222 \
	> ../analysis/run/lcb/sonnet/delegated.log 2>&1 &
