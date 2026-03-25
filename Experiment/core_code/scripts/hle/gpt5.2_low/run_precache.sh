#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 nohup python /data3/peijia/dr-claw/Explain/Experiment/core_code/precache_explores.py \
	--backend codex \
	--cache-dirs ../analysis/cache/hle/gpt5.2_low/gold \
	--subset gold \
	--num 200 \
	--num-explores 8 \
	--num-workers 8 \
	--seed 42 \
	--text-only \
	--explore-model gpt-5.2 \
	--effort low \
	> ../analysis/run/hle/gpt5.2_low/precache.log 2>&1 &
