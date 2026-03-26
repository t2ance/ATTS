#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/gpt5.4_no_integrate
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--backend codex \
	--cache-dirs ../analysis/cache/hle/gpt5.4/gold \
	--subset gold \
	--num 200 \
	--num-explores 8 \
	--num-workers 4 \
	--seed 42 \
	--text-only \
	--explore-model gpt-5.4 \
	> ../analysis/run/hle/gpt5.4_no_integrate/precache.log 2>&1 &
