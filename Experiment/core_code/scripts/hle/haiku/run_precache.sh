#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 nohup python /data3/peijia/dr-claw/Explain/Experiment/core_code/precache_explores.py \
	--backend claude \
	--cache-dirs ../analysis/cache/hle/haiku/gold \
	--subset gold \
	--num 200 \
	--num-explores 8 \
	--num-workers 8 \
	--seed 42 \
	--text-only \
	--explore-model claude-haiku-4-5-20251001 \
	> ../analysis/run/hle/haiku/precache.log 2>&1 &
