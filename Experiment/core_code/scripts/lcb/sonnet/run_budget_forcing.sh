#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark lcb \
	--backend claude \
	--method budget-forcing \
	--seed 42 \
	--num-explores 8 \
	--num-workers 4 \
	--log-dir ../analysis/run/lcb/sonnet_budget_forcing \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--no-cache-only \
	--cache-dirs ../analysis/cache/lcb/sonnet_budget_forcing \
	--resume ../analysis/run/lcb/sonnet_budget_forcing/run_20260314_181955 \
	> ../analysis/run/lcb/sonnet_budget_forcing/budget_forcing.log 2>&1 &
