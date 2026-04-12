#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 python eval.py --benchmark hle \
	--backend vllm \
	--seed 42 \
	--num-explores 8 \
	--num-workers 3 \
	--num 100 \
	--subset gold \
	--text-only \
	--log-dir ../analysis/run/hle/grpo_8b_step78 \
	--orchestrator-model grpo-8b \
	--explore-model grpo-8b \
	--integrate-model grpo-8b \
	--cache-dirs ../analysis/cache/hle/sonnet/gold \
	--no-integrate \
	--resume ../analysis/run/hle/grpo_8b_step78/run_20260411_011423 \
	2>&1 | tee tmp/eval_grpo_8b_hle_resume.log
