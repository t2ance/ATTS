#!/usr/bin/env bash
set -euo pipefail

cd /data3/peijia/dr-claw/Explain/Experiment

python3 -c "
from huggingface_hub import HfApi

api = HfApi()
repo_id = 't2ance/tts-agent-playground'

api.create_repo(repo_id, repo_type='dataset', private=True, exist_ok=True)

api.upload_large_folder(
    folder_path='analysis',
    repo_id=repo_id,
    repo_type='dataset',
)

print('Synced analysis/ to https://huggingface.co/datasets/' + repo_id)
"
