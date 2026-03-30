#!/usr/bin/env bash
set -euo pipefail

# Test the training data pipeline with 3 questions.
# Step 1: Pre-cache 3 Haiku explores
# Step 2: Generate 3 ATTS trajectories
# Step 3: Parse into SFT format

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

unset CLAUDECODE 2>/dev/null || true

echo "Step 1: Pre-caching Haiku explores for 3 questions..."
python precache_explores.py \
    --backend claude \
    --cache-dirs ../analysis/cache/hle/haiku_test/gold \
    --subset gold \
    --num 3 \
    --num-explores 8 \
    --num-workers 3 \
    --seed 42 \
    --text-only \
    --explore-model claude-haiku-4-5-20251001 \
    --effort low

echo "Step 2: Generating ATTS trajectories for 3 questions..."
python eval.py --benchmark hle \
    --backend claude \
    --method tts-agent \
    --subset gold \
    --num 3 \
    --seed 42 \
    --num-explores 8 \
    --num-workers 3 \
    --text-only \
    --log-dir ../analysis/run/hle/training_test \
    --orchestrator-model claude-sonnet-4-6 \
    --explore-model claude-haiku-4-5-20251001 \
    --cache-dirs ../analysis/cache/hle/haiku_test/gold \
    --cache-only

echo "Step 3: Parsing trajectories into SFT format..."
python -m training.build_sft_data

echo "Done. Check training_data/sft_all.jsonl"
