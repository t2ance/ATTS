#!/usr/bin/env bash
set -euo pipefail

# Pre-generate 8 Sonnet explorer rollouts per question for the Counting subset
# of R-Bench-V (195 questions). Cache feeds run_delegated.sh and the
# downstream method runs (self_refine, budget_forcing, visualprm_rerank,
# standalone_integrator) which read these explores cache-only.
#
# num_workers=1 is set in the YAML — see rationale in run_precache_physics.sh.
# When running standalone, override via:
#   conda run ... python precache_explores.py --config <yaml> -o num_workers=8

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/sonnet

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/rbenchv/sonnet/rbenchv_sonnet_precache_counting.yaml \
	>> ../analysis/run/rbenchv/sonnet/precache_counting.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet/precache_counting.log"
