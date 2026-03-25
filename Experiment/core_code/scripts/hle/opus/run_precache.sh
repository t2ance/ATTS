#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 nohup python /data3/peijia/dr-claw/Explain/Experiment/core_code/precache_explores.py \
	--backend claude \
	--cache-dirs ../analysis/cache/hle/opus/gold \
	--subset gold \
	--num 100 \
	--num-explores 8 \
	--num-workers 10 \
	--seed 42 \
	--text-only \
	--explore-model claude-opus-4-6 \
	--effort low \
	> ../analysis/run/hle/opus/precache.log 2>&1 &
