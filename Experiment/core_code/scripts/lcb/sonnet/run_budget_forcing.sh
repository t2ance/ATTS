#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/lcb_sonnet_budget_forcing.yaml \
	-o resume=../analysis/run/lcb/sonnet_budget_forcing/run_20260314_181955 \
	> ../analysis/run/lcb/sonnet_budget_forcing/budget_forcing.log 2>&1 &
