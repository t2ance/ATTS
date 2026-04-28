#!/usr/bin/env bash
set -euo pipefail

# Pre-generate 8 Sonnet explorer rollouts per question for the Physics subset
# of R-Bench-V (157 questions). Cache feeds run_delegated_physics.sh (ATTS)
# which runs cache-only and reads these explores.
#
# num-workers=1 (2026-04-28 instruction): when this precache runs concurrently
# with the LCB/GPQA/BabyVision socratic-SR jobs (all sharing the same Anthropic
# account), 8 workers here + 1 each on the other 3 = ~11 concurrent calls,
# which hit the per-account rate_limit and killed all four jobs simultaneously
# at 08:53 UTC (resets 12:30 UTC). Reducing this to 1 keeps total concurrency
# at 4 across all parallel jobs. If running this script standalone (no other
# socratic-SR jobs in flight), 8 is fine — restore it then.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/sonnet

PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--benchmark rbenchv \
	--backend claude \
	--category Physics \
	--cache-dirs ../analysis/cache/rbenchv/sonnet \
	--num-explores 8 \
	--num-workers 1 \
	--seed 42 \
	--explore-model claude-sonnet-4-6 \
	>> ../analysis/run/rbenchv/sonnet/precache_physics.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet/precache_physics.log"
