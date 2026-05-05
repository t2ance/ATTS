#!/usr/bin/env bash
set -euo pipefail

# Pre-cache 8 explore rollouts per qid for R-Bench-V Physics (157 Qs) on
# google/gemini-3-flash-preview via OpenRouter. precache is cache-aware
# (file-existence check at precache_explores.py:99); re-launch is safe.
#
# OpenRouter API key sourced via grep+eval (CLAUDE.md API-key-freshness
# protocol): long-lived parent shells fixate env at startup, so directly
# trusting $OPENROUTER_API_KEY can route to a stale key and silently 401.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Physics

eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
    --config scripts/rbenchv/openrouter/rbenchv_gemini-3-flash-preview_precache_physics.yaml \
    >> ../analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Physics/precache.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Physics/precache.log"
