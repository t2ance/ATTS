#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--benchmark babyvision \
	--backend claude \
	--cache-dirs ../analysis/cache/babyvision/opus \
	--num-explores 2 \
	--num-workers 8 \
	--seed 42 \
	--explore-model claude-opus-4-6 \
	> ../analysis/run/babyvision/opus/precache.log 2>&1 &
