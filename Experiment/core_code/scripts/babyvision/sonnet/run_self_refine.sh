#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/babyvision_sonnet_self_refine.yaml \
	-o resume=../analysis/run/babyvision/sonnet_self_refine/run_20260313_231138 \
	> ../analysis/run/babyvision/sonnet_self_refine/self_refine.log 2>&1 &
