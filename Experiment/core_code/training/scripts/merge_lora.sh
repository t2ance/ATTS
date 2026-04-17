set -x

ADAPTER_PATH="/data3/peijia/dr-claw/Explain/Experiment/core_code/checkpoints/sft_qwen3_8b"
OUTPUT_PATH="/data3/peijia/dr-claw/Explain/Experiment/core_code/checkpoints/sft_qwen3_8b_merged"

conda run --no-capture-output -n grpo python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_name = 'Qwen/Qwen3-8B'
adapter_path = '$ADAPTER_PATH'
output_path = '$OUTPUT_PATH'

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(model, adapter_path)

print('Merging...')
model = model.merge_and_unload()

print(f'Saving merged model to {output_path}...')
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print('Done')
"
