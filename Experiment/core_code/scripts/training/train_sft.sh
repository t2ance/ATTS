#!/bin/bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n verl llamafactory-cli train training/sft_config.yaml
