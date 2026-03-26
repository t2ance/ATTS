#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/gpqa/gpt5.2_no_integrate
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--benchmark gpqa \
	--backend codex \
	--cache-dirs ../analysis/cache/gpqa/gpt5.2 \
	--num-explores 8 \
	--num-workers 4 \
	--seed 42 \
	--explore-model gpt-5.2 \
	--effort low \
	> ../analysis/run/gpqa/gpt5.2_no_integrate/precache.log 2>&1 &
