#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# -o resume=<RUN_DIR> stays as a CLI flag (per-launch, not per-config) so the
# YAML can be reused across rerun attempts pointing at different run dirs.
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/lcb/multi_model/lcb_multi_delegated.yaml \
	-o resume=../analysis/run/lcb/multi_model/run_20260320_220841 \
	> ../analysis/run/lcb/multi_model/delegated_resume.log 2>&1 &
