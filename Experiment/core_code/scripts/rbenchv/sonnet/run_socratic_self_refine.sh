#!/usr/bin/env bash
set -euo pipefail

# socratic-self-refine on R-Bench-V Physics subset (157 questions, smallest of 4
# categories — Game 275, Counting 195, Math 176, Physics 157). Sonnet across all
# three roles to mirror sonnet_socratic_self_refine on HLE.
#
# num-workers=1 per the 2026-04-28 instruction to keep parallelism low across
# concurrently running socratic-SR jobs (HLE/LCB/GPQA/BabyVision/RBenchV share
# the same Anthropic account quota). Earlier 4/8-worker runs hit rate limits.
#
# --no-cache-only: this method generates and caches explores in the same pass,
# so a separate precache step is not required for it.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/sonnet_socratic_self_refine

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/rbenchv/sonnet/rbenchv_sonnet_socratic_self_refine.yaml \
	>> ../analysis/run/rbenchv/sonnet_socratic_self_refine/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_socratic_self_refine/socratic_self_refine.log"
