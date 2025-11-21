#!/bin/bash
# conda activate step_dpo

# Single-anget
DATA_PATH="data_pipeline/pref_pairs_augmented/preference_pairs-Mistral-7B-Instruct-v0.3-10.json"
# DATA_PATH="data_pipeline/pref_pairs"

MODEL_NAME_OR_PATH=mistralai/Mistral-7B-Instruct-v0.3
# MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3-8B-Instruct
# MODEL_NAME_OR_PATH="google/gemma-3-1b-it"
# MODEL_NAME_OR_PATH=Qwen/Qwen2-72B-Instruct
# MODEL_NAME_OR_PATH=Qwen/Qwen2-0.5B-Instruct
MODEL_ID=Mistral-7B-Instruct-v0.3-dpo-preference_pairs-Mistral-7B-Instruct-v0.3-10


ACCELERATE_LOG_LEVEL=info accelerate launch --gpu_ids 0 --config_file accelerate_configs/deepspeed_zero3_cpu.yaml --mixed_precision bf16 \
    --num_processes 1 \
    train.py \
    --do_train \
    --eval_strategy 'steps' \
    --eval_steps 5 \
    --config configs/config_full.yaml \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_path ${DATA_PATH} \
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
    # --prompt=qwen2-boxed
    # --do_eval \
    # --num_train_epochs=4 \
# python train.py configs/config_full.yaml \