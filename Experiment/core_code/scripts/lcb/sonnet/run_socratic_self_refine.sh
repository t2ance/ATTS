#!/usr/bin/env bash
set -euo pipefail

# Socratic-Self-Refine on LiveCodeBench. num-workers=1 per the 2026-04-28
# instruction to keep parallelism low across concurrently running socratic-SR
# jobs (HLE/LCB/GPQA/BabyVision share the same Anthropic account quota).
#
# --no-cache-only: this method generates and caches explores in the same pass.
#
# --resume: question-level resume from prior run's results.jsonl. Reuses
# explore cache + skips already-graded questions. The chosen run_dir is the
# one with the largest results.jsonl on 2026-04-28; if you start a fresh
# experiment (different seed/method/model), bump or remove this path.

unset CLAUDECODE 2>/dev/null || true

# benchmarks/grader.py:185 imports lcb_runner.evaluation.compute_code_generation_metrics
# for code grading. lcb_runner is the local LiveCodeBench checkout (not pip-installed),
# so it must be on PYTHONPATH. The older run_self_refine.sh (2026-03-14) ran inside
# an interactive shell that had this exported; explicit launchers must set it.
export PYTHONPATH=/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench:${PYTHONPATH:-}

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

mkdir -p ../analysis/run/lcb/sonnet_socratic_self_refine

PYTHONUNBUFFERED=1 nohup python eval.py --benchmark lcb \
	--backend claude \
	--method socratic-self-refine \
	--seed 42 \
	--num-explores 8 \
	--num-workers 1 \
	--log-dir ../analysis/run/lcb/sonnet_socratic_self_refine \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--no-cache-only \
	--cache-dirs ../analysis/cache/lcb/sonnet_socratic_self_refine \
	--resume ../analysis/run/lcb/sonnet_socratic_self_refine/run_20260428_163748 \
	>> ../analysis/run/lcb/sonnet_socratic_self_refine/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_socratic_self_refine/socratic_self_refine.log"
