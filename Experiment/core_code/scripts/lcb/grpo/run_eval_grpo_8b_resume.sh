#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
export PYTHONPATH="/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench:${PYTHONPATH:-}"

PYTHONUNBUFFERED=1 python eval.py --benchmark lcb \
	--backend vllm \
	--seed 42 \
	--num-explores 8 \
	--num-workers 3 \
	--log-dir ../analysis/run/lcb/grpo_8b_step78 \
	--orchestrator-model grpo-8b \
	--explore-model grpo-8b \
	--integrate-model grpo-8b \
	--cache-dirs ../analysis/cache/lcb/sonnet \
	--no-integrate \
	--resume ../analysis/run/lcb/grpo_8b_step78/run_20260410_210724 \
	2>&1 | tee tmp/eval_grpo_8b_lcb_resume.log
