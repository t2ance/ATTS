#!/usr/bin/env bash
set -euo pipefail

# Pre-generate 4 Sonnet explorer rollouts per question for the Physics subset
# of R-Bench-V (157 questions). Cache feeds run_delegated_physics.sh (ATTS)
# which runs cache-only and reads these explores.
#
# 2026-05-04: switched 8->4 explores per qid to halve remaining budget.
# precache is cache-aware (file-existence check at precache_explores.py:99),
# so the 62 qids already at >=4 real explores skip with 0 spend; only 378
# new explores fire (94 untouched x 4 + physics_63 2/4 -> +2). See yaml for
# detail. The 5-8-slot files for fully-cached qids stay on disk untouched.
#
# num-workers=4 (yaml): single-job standalone, no rate-limit collision risk.
# If you re-launch concurrently with other Anthropic-backend jobs, drop to 1.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/sonnet

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/rbenchv/sonnet/rbenchv_sonnet_precache_physics.yaml \
	>> ../analysis/run/rbenchv/sonnet/precache_physics.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet/precache_physics.log"
