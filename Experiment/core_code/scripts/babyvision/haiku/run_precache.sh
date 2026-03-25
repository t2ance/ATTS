#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--benchmark babyvision \
	--backend claude \
	--cache-dirs ../analysis/cache/babyvision/haiku \
	--num-explores 8 \
	--num-workers 8 \
	--seed 42 \
	--explore-model claude-haiku-4-5-20251001 \
	> ../analysis/run/babyvision/haiku/precache.log 2>&1 &
