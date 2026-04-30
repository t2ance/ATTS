#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/sonnet/gpqa_sonnet_skywork_rerank.yaml \
	> ../analysis/run/gpqa/sonnet_skywork_rerank/rerank.log 2>&1 &
