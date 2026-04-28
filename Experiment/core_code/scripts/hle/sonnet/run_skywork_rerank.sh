#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude \
	--method rerank \
	--subset gold \
	--text-only \
	--num 100 \
	--reward-model Skywork/Skywork-Reward-V2-Qwen3-8B \
	--seed 42 \
	--num-explores 8 \
	--num-workers 16 \
	--log-dir ../analysis/run/hle/sonnet_skywork_rerank \
	--explore-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/hle/sonnet/gold \
	> ../analysis/run/hle/sonnet_skywork_rerank/rerank.log 2>&1 &
