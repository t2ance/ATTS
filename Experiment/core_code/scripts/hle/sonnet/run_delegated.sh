#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/hle/sonnet/hle_sonnet_delegated.yaml \
	-o resume=../analysis/run/hle/sonnet/run_20260306_170321 \
	> ../analysis/run/hle/sonnet/gold_delegated.log 2>&1 &
