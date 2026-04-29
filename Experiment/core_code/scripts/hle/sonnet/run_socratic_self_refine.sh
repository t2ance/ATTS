#!/usr/bin/env bash
set -euo pipefail

# Resumes the in-progress run; --resume skips qids already in results.jsonl.
# num-workers=1 so that an Anthropic Usage Policy refusal on one qid cannot
# affect concurrent qids, making crash attribution unambiguous. Policy retry
# is handled in backends/claude.py (_POLICY_MAX_RETRIES=2, 30s delay).
#
# judge_model NOT set in YAML: hle.py class default (claude-haiku-4-5-20251001)
# is the source of truth. 2026-04-28 incident: hle.py had judge_model=None as
# a TEMP smoke-test override + this launcher had no flag -> all 100 grade.json
# wrote judge_model="none" -> string-match grading -> SSR acc -8pp underestimate.
# After hle.py fix, launcher does not need to override. To override, add to YAML:
#   judge_model: <model>

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config scripts/hle/sonnet/hle_sonnet_socratic_self_refine.yaml \
	-o resume=../analysis/run/hle/sonnet_socratic_self_refine/run_20260427_071039 \
	>> ../analysis/run/hle/sonnet_socratic_self_refine/socratic_self_refine.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_socratic_self_refine/socratic_self_refine.log"
