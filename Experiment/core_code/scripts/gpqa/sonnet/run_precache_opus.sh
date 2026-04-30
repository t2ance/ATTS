#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/gpqa/sonnet/gpqa_sonnet_precache_opus.yaml \
	> ../analysis/cache/gpqa/opus_precache.log 2>&1 &
