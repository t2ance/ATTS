#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# -o resume=<RUN_DIR> stays as a CLI flag (per-launch, not per-config) so the
# YAML can be reused across rerun attempts pointing at different run dirs.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/hle/multi_model/hle_multi_delegated.yaml \
	-o resume=../analysis/run/hle/multi_model/run_20260320_183005 \
	> ../analysis/run/hle/multi_model/delegated_rerun.log 2>&1 &
