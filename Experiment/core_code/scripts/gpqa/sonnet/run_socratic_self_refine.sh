#!/usr/bin/env bash
set -euo pipefail

# Socratic-Self-Refine on GPQA-Diamond. num-workers=1 per the 2026-04-28
# instruction to keep parallelism low across concurrently running socratic-SR
# jobs (HLE/LCB/GPQA/BabyVision share the same Anthropic account quota).
#
# --no-cache-only: this method generates and caches explores in the same pass.
#
# --resume: question-level resume from prior run's results.jsonl. Reuses
# explore cache + skips already-graded questions. The chosen run_dir is the
# one with the largest results.jsonl on 2026-04-28; if you start a fresh
# experiment (different seed/method/model), bump or remove this path.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/gpqa/sonnet_socratic_self_refine

# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/sonnet/gpqa_sonnet_socratic_self_refine.yaml \
	-o resume=../analysis/run/gpqa/sonnet_socratic_self_refine/run_20260428_163751 \
	>> ../analysis/run/gpqa/sonnet_socratic_self_refine/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_socratic_self_refine/socratic_self_refine.log"
