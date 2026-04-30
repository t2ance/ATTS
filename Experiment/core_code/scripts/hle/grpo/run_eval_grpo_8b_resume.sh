#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/grpo/hle_grpo_8b_resume.yaml \
	-o resume=../analysis/run/hle/grpo_8b_step78/run_20260411_011423 \
	2>&1 | tee tmp/eval_grpo_8b_hle_resume.log
