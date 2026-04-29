#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
export PYTHONPATH="/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench:${PYTHONPATH:-}"

PYTHONUNBUFFERED=1 python eval.py \
	--config scripts/lcb/grpo/lcb_grpo_8b.yaml \
	2>&1 | tee tmp/eval_grpo_8b_lcb.log
