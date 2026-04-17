set -x
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run --no-capture-output -n verl llamafactory-cli train training/sft_config_qwen3_8b.yaml
