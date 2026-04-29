#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/gpqa_multi_delegated_v2.yaml \
	> ../analysis/run/gpqa/multi_model_v2/delegated.log 2>&1 &
