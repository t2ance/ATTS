#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/sonnet/gpqa_sonnet_budget_forcing.yaml \
	-o resume=../analysis/run/gpqa/sonnet_budget_forcing/run_20260313_230055 \
	> ../analysis/run/gpqa/sonnet_budget_forcing/budget_forcing.log 2>&1 &
