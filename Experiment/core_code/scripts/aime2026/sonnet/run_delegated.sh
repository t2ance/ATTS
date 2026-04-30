#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/aime2026/sonnet/aime2026_sonnet_delegated.yaml \
	-o resume=../analysis/run/aime2026/sonnet/run_20260308_161605 \
	> ../analysis/run/aime2026/sonnet/delegated.log 2>&1 &
