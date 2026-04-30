#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/babyvision/sonnet/babyvision_sonnet_visualprm_rerank.yaml \
	-o resume=../analysis/run/babyvision/sonnet_visualprm_rerank/run_20260315_153901 \
	> ../analysis/run/babyvision/sonnet_visualprm_rerank/rerank.log 2>&1 &
