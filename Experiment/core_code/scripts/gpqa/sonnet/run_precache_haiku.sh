#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
PYTHONUNBUFFERED=1 nohup python precache_explores.py --benchmark gpqa \
	--backend claude \
	--seed 42 \
	--num-explores 8 \
	--num-workers 8 \
	--explore-model claude-haiku-4-5-20251001 \
	--cache-dirs ../analysis/cache/gpqa/haiku \
	> ../analysis/cache/gpqa/haiku_precache.log 2>&1 &
