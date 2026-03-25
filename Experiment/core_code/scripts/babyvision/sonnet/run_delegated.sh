#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark babyvision \
	--backend claude \
	--seed 42 \
	--num-explores 8 \
	--num-workers 4 \
	--log-dir ../analysis/run/babyvision/sonnet \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/babyvision/sonnet \
	> ../analysis/run/babyvision/sonnet/delegated.log 2>&1 &
