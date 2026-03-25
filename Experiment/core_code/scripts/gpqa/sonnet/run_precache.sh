#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--benchmark gpqa \
	--backend claude \
	--cache-dirs ../analysis/cache/gpqa/sonnet \
	--num-explores 8 \
	--num-workers 4 \
	--seed 42 \
	--explore-model claude-sonnet-4-6 \
	> ../analysis/run/gpqa/sonnet/precache.log 2>&1 &
