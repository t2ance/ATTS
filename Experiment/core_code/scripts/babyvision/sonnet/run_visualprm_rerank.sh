#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/babyvision/sonnet/babyvision_sonnet_visualprm_rerank.yaml \
	> ../analysis/run/babyvision/sonnet_visualprm_rerank/rerank.log 2>&1 &
