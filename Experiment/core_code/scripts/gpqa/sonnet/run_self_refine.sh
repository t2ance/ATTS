#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/gpqa_sonnet_self_refine.yaml \
	-o resume=../analysis/run/gpqa/sonnet_self_refine/run_20260313_230052 \
	> ../analysis/run/gpqa/sonnet_self_refine/self_refine.log 2>&1 &
