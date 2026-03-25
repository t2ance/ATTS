#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark lcb \
	--backend claude \
	--num 10 \
	--seed 42 \
	--num-explores 8 \
	--num-workers 16 \
	--log-dir ../analysis/run/lcb/haiku \
	--orchestrator-model claude-haiku-4-5-20251001 \
	--explore-model claude-haiku-4-5-20251001 \
	--integrate-model claude-haiku-4-5-20251001 \
	--cache-dirs ../analysis/cache/lcb/haiku \
	> ../analysis/run/lcb/haiku/delegated.log 2>&1 &
