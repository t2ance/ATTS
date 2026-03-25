#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude \
	--subset gold \
	--num 100 \
	--seed 42 \
	--num-explores 8 \
	--num-workers 16 \
	--text-only \
	--log-dir ../analysis/run/hle/sonnet \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/hle/sonnet/gold \
	--resume ../analysis/run/hle/sonnet/run_20260306_170321 \
	> ../analysis/run/hle/sonnet/gold_delegated.log 2>&1 &
