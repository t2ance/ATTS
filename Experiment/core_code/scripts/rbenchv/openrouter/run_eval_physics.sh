#!/usr/bin/env bash
set -euo pipefail

# ATTS evaluation on R-Bench-V Physics (157 Qs) using gemini-3-flash-preview.
# Reads from the cache produced by run_precache_physics.sh.
#
# OpenRouter API key sourced via grep+eval (CLAUDE.md API-key-freshness
# protocol).

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Physics

eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config scripts/rbenchv/openrouter/rbenchv_gemini-3-flash-preview_eval_physics.yaml \
    > ../analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Physics/eval.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Physics/eval.log"
