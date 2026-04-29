#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/aime2025/sonnet/aime2025_sonnet_skywork_rerank.yaml \
	> ../analysis/run/aime2025/sonnet_skywork_rerank/rerank.log 2>&1 &
