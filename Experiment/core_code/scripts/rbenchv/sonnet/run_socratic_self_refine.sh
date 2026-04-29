#!/usr/bin/env bash
set -euo pipefail

# socratic-self-refine on R-Bench-V Physics subset (157 questions, smallest of 4
# categories — Game 275, Counting 195, Math 176, Physics 157). Sonnet across all
# three roles to mirror sonnet_socratic_self_refine on HLE.
#
# num-workers=4: matches HLE setup; lower than precache's 8 to keep Anthropic
# quota burst under the per-account rate-limit ceiling. Earlier 8-worker runs on
# HLE (2026-04-28) hit "You've hit your limit" mid-run.
#
# --no-cache-only: this method generates and caches explores in the same pass,
# so a separate precache step is not required for it.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/sonnet_socratic_self_refine

PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/rbenchv_sonnet_socratic_self_refine.yaml \
	>> ../analysis/run/rbenchv/sonnet_socratic_self_refine/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_socratic_self_refine/socratic_self_refine.log"
