#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
export PYTHONPATH="/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench:${PYTHONPATH:-}"

# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 python eval.py \
	--config scripts/lcb/grpo/lcb_grpo_8b.yaml \
	-o resume=../analysis/run/lcb/grpo_8b_step78/run_20260410_210724 \
	2>&1 | tee tmp/eval_grpo_8b_lcb_resume.log
