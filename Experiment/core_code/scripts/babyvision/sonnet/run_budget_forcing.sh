#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/babyvision/sonnet/babyvision_sonnet_budget_forcing.yaml \
	-o resume=../analysis/run/babyvision/sonnet_budget_forcing/run_20260313_231131 \
	> ../analysis/run/babyvision/sonnet_budget_forcing/budget_forcing.log 2>&1 &
