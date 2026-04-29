#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/aime2025_sonnet_delegated.yaml \
	-o resume=../analysis/run/aime2025/sonnet/run_20260308_161617 \
	> ../analysis/run/aime2025/sonnet/delegated.log 2>&1 &
