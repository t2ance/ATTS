#!/usr/bin/env bash
set -euo pipefail

# Resumes the in-progress run; --resume skips qids already in results.jsonl.
# num-workers=1 so that an Anthropic Usage Policy refusal on one qid cannot
# affect concurrent qids, making crash attribution unambiguous. Policy retry
# is handled in backends/claude.py (_POLICY_MAX_RETRIES=2, 30s delay).

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude \
	--method socratic-self-refine \
	--subset gold \
	--num 100 \
	--text-only \
	--seed 42 \
	--num-explores 8 \
	--num-workers 1 \
	--resume ../analysis/run/hle/sonnet_socratic_self_refine/run_20260427_071039 \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--no-cache-only \
	--cache-dirs ../analysis/cache/hle/sonnet_socratic_self_refine/gold \
	>> ../analysis/run/hle/sonnet_socratic_self_refine/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_socratic_self_refine/socratic_self_refine.log"
