#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python eval.py --benchmark rbenchv \
	--backend claude \
	--method rerank \
	--reward-model OpenGVLab/VisualPRM-8B \
	--seed 42 \
	--num-explores 8 \
	--num-workers 16 \
	--log-dir ../analysis/run/rbenchv/sonnet_visualprm_rerank \
	--explore-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/rbenchv/sonnet \
	> ../analysis/run/rbenchv/sonnet_visualprm_rerank/rerank.log 2>&1 &
