#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark gpqa \
	--backend claude \
	--method budget-forcing \
	--seed 42 \
	--num-explores 8 \
	--num-workers 16 \
	--log-dir ../analysis/run/gpqa/sonnet_budget_forcing \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--no-cache-only \
	--cache-dirs ../analysis/cache/gpqa/sonnet_budget_forcing \
	--resume ../analysis/run/gpqa/sonnet_budget_forcing/run_20260313_230055 \
	> ../analysis/run/gpqa/sonnet_budget_forcing/budget_forcing.log 2>&1 &
