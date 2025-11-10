#!/bin/bash

# Switch from conda env privacylens to step_dpo:
# conda activate step_dpo

# Single-anget

MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
MODEL_ID=Llama-3.1-8B-Instruct-dpo-lora
# MODEL_NAME=Qwen/Qwen2-72B-Instruct
# MODEL_ID=qwen2-72b-instruct-step-dpo
# MODEL_NAME=Qwen/Qwen2-0.5B-Instruct
# MODEL_ID=qwen2-0.5b-instruct-step-dpo
DATA_PATH=data_pipeline/pref_pairs/train.json

# # Multi-agent
# MODEL_ID=qwen2-0.5b-instruct-dpo-refiner
# DATA_PATH=data_pipeline_MA/pref_pairs/pref_pairs_refiner.json

# ACCELERATE_LOG_LEVEL=info accelerate launch --gpu_ids 1,2 --config_file accelerate_configs/deepspeed_zero3_cpu.yaml --mixed_precision bf16 \
#     --num_processes 2 \

CUDA_VISIBLE_DEVICES=4 python train_lora.py \
    --do_train \
    --config configs/config_full.yaml \
    --model_name_or_path=${MODEL_NAME} \
    --data_path=$DATA_PATH \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --beta=0.4 \
    --num_train_epochs=16 \
    --dataloader_num_workers 8 \
    --save_strategy='steps' \
    --save_steps=200 \
    --save_total_limit=1 \
    --output_dir=outputs/${MODEL_ID} \
    --hub_model_id=${MODEL_ID} \
    --prompt=qwen2-boxed
    # --num_train_epochs=4 \
# python train.py configs/config_full.yaml \