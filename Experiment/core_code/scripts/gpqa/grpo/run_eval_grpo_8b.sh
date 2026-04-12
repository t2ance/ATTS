#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 python eval.py --benchmark gpqa \
	--backend vllm \
	--seed 42 \
	--num-explores 8 \
	--num-workers 1 \
	--log-dir ../analysis/run/gpqa/grpo_8b_step78 \
	--orchestrator-model grpo-8b \
	--explore-model grpo-8b \
	--integrate-model grpo-8b \
	--cache-dirs ../analysis/cache/gpqa/sonnet \
	--no-integrate \
	2>&1 | tee tmp/eval_grpo_8b.log
