#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/gpqa/gpt5.2_no_integrate
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/gpt5.2/gpqa_gpt5.2_no_integrate.yaml \
	> ../analysis/run/gpqa/gpt5.2_no_integrate/delegated.log 2>&1 &
