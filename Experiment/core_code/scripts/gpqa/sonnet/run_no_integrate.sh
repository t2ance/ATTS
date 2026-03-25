#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark gpqa \
	--backend claude \
	--seed 42 \
	--num-explores 8 \
	--num-workers 4 \
	--log-dir ../analysis/run/gpqa/sonnet_no_integrate \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/gpqa/sonnet \
	--no-integrate \
	> ../analysis/run/gpqa/sonnet_no_integrate/delegated.log 2>&1 &
