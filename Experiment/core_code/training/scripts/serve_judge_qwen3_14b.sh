set -x
ulimit -n 65535

# Qwen3-14B judge server on GPU 2/3. Replaces 8B judge which hallucinated
# extracted_final_answer = gold on off-topic responses.
export CUDA_VISIBLE_DEVICES=2,3

VERL_ENV="/home/peijia/miniconda3/envs/grpo_vllm"
export LD_LIBRARY_PATH="$VERL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$VERL_ENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

conda run --no-capture-output -n grpo_vllm python3 -u -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --served-model-name judge \
    --host 127.0.0.1 \
    --port 8765 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --no-enable-log-requests \
    2>&1 | tee /data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/judge_qwen3_14b.log
