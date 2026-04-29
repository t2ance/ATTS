#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--config configs/gpqa_sonnet_precache_haiku.yaml \
	> ../analysis/cache/gpqa/haiku_precache.log 2>&1 &
