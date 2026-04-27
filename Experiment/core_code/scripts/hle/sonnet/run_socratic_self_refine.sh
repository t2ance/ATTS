#!/usr/bin/env bash
set -euo pipefail

# Socratic Self-Refine smoke run on HLE Verified (Gold subset, 100 questions).
#
# Mirrors run_self_refine.sh exactly EXCEPT:
#   - --method socratic-self-refine     (uses methods/socratic_self_refine.py)
#   - --cache-dirs ../analysis/cache/hle/sonnet_socratic_self_refine_v5/gold
#       => fresh cache namespace; existing sonnet_self_refine cache is untouched
#   - --log-dir + log file under sonnet_socratic_self_refine_v5
#
# All other knobs (subset, seed, num_explores, num_workers, models) are
# intentionally identical so the only experimental variable is the Critic
# system prompt -- making the resulting num_explores distribution and final
# accuracy directly comparable to the existing sonnet_self_refine results.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/hle/sonnet_socratic_self_refine_v5

# --num-workers 2 -- carried over from v4. v3 @4 crashed with
# `invalid_request: Usage Policy` (after 14/100 vs v1@16's 3/100 -- progressive
# but not eliminated). Dropping to 2 halves classifier-evaluation rate again.
# Fail-fast preserved -- if it still crashes, parallelism is not the root cause
# and the trigger is a specific HLE qid that Sonnet-4.6 safety flags regardless
# of concurrency.
#
# v5 = v4 prompt + injected "## Iteration State" block in build_feedback_message
# so the Critic can actually see the refine-count the procedural floor needs.
# v4 was killed at 1 partial qid after trace inspection showed the Critic
# always wrote "round 1, 0 drafts" because its user message contained zero
# history (Madaan Algorithm 1 line 3 passes only the latest draft). v4 cache
# (`sonnet_socratic_self_refine_v4`) preserved untouched as the broken-floor
# baseline.
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude \
	--method socratic-self-refine \
	--subset gold \
	--num 100 \
	--text-only \
	--seed 42 \
	--num-explores 8 \
	--num-workers 2 \
	--log-dir ../analysis/run/hle/sonnet_socratic_self_refine_v5 \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--no-cache-only \
	--cache-dirs ../analysis/cache/hle/sonnet_socratic_self_refine_v5/gold \
	> ../analysis/run/hle/sonnet_socratic_self_refine_v5/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!  Tail with:"
echo "  tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_socratic_self_refine_v5/socratic_self_refine.log"
