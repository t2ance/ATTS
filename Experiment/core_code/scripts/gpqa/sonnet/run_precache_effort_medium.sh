#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--benchmark gpqa \
	--backend claude \
	--cache-dirs ../analysis/cache/gpqa/sonnet_effort_medium \
	--num-explores 8 \
	--num-workers 1 \
	--seed 42 \
	--explore-model claude-sonnet-4-6 \
	--effort medium \
	--explore-timeout 1200 \
	> ../analysis/run/gpqa/sonnet_effort_medium_precache5.log 2>&1 &
