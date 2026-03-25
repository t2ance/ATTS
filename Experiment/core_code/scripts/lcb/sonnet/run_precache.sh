#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--benchmark lcb \
	--backend claude \
	--cache-dirs ../analysis/cache/lcb/sonnet \
	--num-explores 8 \
	--num-workers 16 \
	--seed 42 \
	--explore-model claude-sonnet-4-6 \
	> ../analysis/run/lcb/sonnet/precache.log 2>&1 &
