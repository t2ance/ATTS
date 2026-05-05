#!/usr/bin/env bash
set -euo pipefail

# ATTS evaluation on R-Bench-V Counting (4-sample smoke) using
# google/gemini-3-flash-preview via OpenRouter. NO separate precache:
# tts-agent defaults to cache_only=False, so the run does cache miss ->
# live API -> writeback automatically. The cache_dir under analysis/cache/
# is empty at launch and gets populated by this run.
#
# OpenRouter API key sourced via grep+eval (CLAUDE.md API-key-freshness
# protocol).

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Counting

eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"

# Pre-flight: verify the OpenRouter key is alive before burning subprocess
# startup time on a 401 (CLAUDE.md API-key-freshness incident 2026-05-04).
HTTP_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
  https://openrouter.ai/api/v1/auth/key)
if [[ "${HTTP_STATUS}" != "200" ]]; then
    echo "OpenRouter pre-flight FAILED: HTTP=${HTTP_STATUS}. Aborting." >&2
    exit 1
fi
echo "OpenRouter pre-flight OK: HTTP=${HTTP_STATUS}"

PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config scripts/rbenchv/openrouter/rbenchv_gemini-3-flash-preview_eval_counting.yaml \
    > ../analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Counting/eval.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/openrouter_gemini-3-flash-preview/Counting/eval.log"
