#!/bin/bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0 /home/peijia/miniconda3/envs/verl/bin/vllm serve Qwen/Qwen3-8B \
    --enable-lora --max-lora-rank 64 \
    --lora-modules atts-orch=checkpoints/sft_qwen3_8b \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 32768 --port 8000 --dtype bfloat16 --trust-remote-code
