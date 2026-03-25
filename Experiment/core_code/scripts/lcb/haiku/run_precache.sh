#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

PYTHONUNBUFFERED=1 nohup python /data3/peijia/dr-claw/Explain/Experiment/core_code/precache_explores.py \
	--benchmark lcb \
	--backend claude \
	--cache-dirs ../analysis/cache/lcb/haiku \
	--num 100 \
	--num-explores 8 \
	--num-workers 16 \
	--seed 42 \
	--explore-model claude-haiku-4-5-20251001 \
	> ../analysis/run/lcb/haiku/precache.log 2>&1 &
