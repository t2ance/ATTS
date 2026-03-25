#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup env -u CLAUDECODE python precache_explores.py \
	--benchmark lcb \
	--backend claude \
	--cache-dirs ../analysis/cache/lcb/opus \
	--num-explores 4 \
	--num-workers 8 \
	--seed 42 \
	--explore-model claude-opus-4-6 \
	> ../analysis/cache/lcb/opus_precache.log 2>&1 &
