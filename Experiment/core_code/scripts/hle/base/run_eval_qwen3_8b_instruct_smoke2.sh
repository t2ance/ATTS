#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 python eval.py \
	--config scripts/hle/base/hle_qwen3_8b_instruct_smoke2.yaml \
	2>&1 | tee tmp/eval_qwen3_8b_instruct_smoke2.log
