#!/bin/bash

# Switch from conda env privacylens to step_dpo:
# conda activate step_dpo


MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
MODEL_ID=Llama-3.1-8B-Instruct-dpo-lora
DATA_PATH=data_pipeline/pref_pairs/train.json


CUDA_VISIBLE_DEVICES=4 python get_llm_embedding.py \
    --model_name_or_path=${MODEL_NAME} \
    --data_path=$DATA_PATH \
    # --num_points 10
    # --torch_dtype=bfloat16 \
    # --bf16=True \