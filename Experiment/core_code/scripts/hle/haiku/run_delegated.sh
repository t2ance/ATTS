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
	--log-dir ../analysis/run/hle/haiku \
	--orchestrator-model claude-haiku-4-5-20251001 \
	--explore-model claude-haiku-4-5-20251001 \
	--integrate-model claude-haiku-4-5-20251001 \
	--cache-dirs ../analysis/cache/hle/haiku/gold \
	> ../analysis/run/hle/haiku/gold_delegated.log 2>&1 &
