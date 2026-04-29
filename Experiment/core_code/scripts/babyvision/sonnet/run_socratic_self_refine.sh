#!/usr/bin/env bash
set -euo pipefail

# Socratic-Self-Refine on BabyVision. num-workers=1 per the 2026-04-28
# instruction to keep parallelism low across concurrently running socratic-SR
# jobs (HLE/LCB/GPQA/BabyVision share the same Anthropic account quota).
#
# --no-cache-only: this method generates and caches explores in the same pass.
#
# --resume: question-level resume from prior run's results.jsonl. Reuses
# explore cache + skips already-graded questions. The chosen run_dir is the
# 08:47 run (13 records), NOT the more recent 16:38 run (only 4 records) —
# pick the run_dir with the LARGEST results.jsonl, not the latest. If you
# start a fresh experiment (different seed/method/model), bump or remove.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/babyvision/sonnet_socratic_self_refine

# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/babyvision/sonnet/babyvision_sonnet_socratic_self_refine.yaml \
	-o resume=../analysis/run/babyvision/sonnet_socratic_self_refine/run_20260428_075116 \
	>> ../analysis/run/babyvision/sonnet_socratic_self_refine/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/babyvision/sonnet_socratic_self_refine/socratic_self_refine.log"
