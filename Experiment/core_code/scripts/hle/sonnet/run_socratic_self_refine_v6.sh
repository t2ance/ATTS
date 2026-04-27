#!/usr/bin/env bash
set -euo pipefail

# v6 = v5 + two changes:
#   - --num-workers 1  (v5@2 crashed at 16/100 with Usage Policy; isolating to 1 worker
#                       ensures a policy refusal on one qid cannot affect a concurrent qid
#                       and makes crash attribution unambiguous)
#   - --resume v5 run dir  (skips the 16 questions already completed in v5)
#
# Policy retry is now handled in backends/claude.py (_POLICY_MAX_RETRIES=2, 30s delay).
# If the refusal is still deterministic after 2 retries, the run will crash on that qid
# so we can inspect it directly.
#
# Cache namespace stays sonnet_socratic_self_refine_v5/gold so no API calls are
# re-issued for already-completed qids.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/hle/sonnet_socratic_self_refine_v5

PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude \
	--method socratic-self-refine \
	--subset gold \
	--num 100 \
	--text-only \
	--seed 42 \
	--num-explores 8 \
	--num-workers 1 \
	--resume ../analysis/run/hle/sonnet_socratic_self_refine_v5/run_20260427_071039 \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--no-cache-only \
	--cache-dirs ../analysis/cache/hle/sonnet_socratic_self_refine_v5/gold \
	> ../analysis/run/hle/sonnet_socratic_self_refine_v5/socratic_self_refine_v6.log 2>&1 &

echo "Launched. PID=$!  Tail with:"
echo "  tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_socratic_self_refine_v5/socratic_self_refine_v6.log"
