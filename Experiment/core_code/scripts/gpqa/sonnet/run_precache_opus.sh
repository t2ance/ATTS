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
	--explore-model claude-opus-4-6 \
	--cache-dirs ../analysis/cache/gpqa/opus \
	> ../analysis/cache/gpqa/opus_precache.log 2>&1 &
