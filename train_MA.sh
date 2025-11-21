#!/bin/bash


# Multi-agent
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ROLE>"
    echo "ROLE must be either 'verifier' or 'refiner'"
    exit 1
fi

ROLE="$1"
if [[ "${ROLE}" != "verifier" && "${ROLE}" != "refiner" ]]; then
    echo "Error: ROLE must be either 'verifier' or 'refiner'."
    exit 1
fi

DATA_PATH=data_pipeline_MA/predictions/Mistral-7B-Instruct-v0.2-branch_4-pref_pairs_${ROLE}.json
MODEL_NAME_OR_PATH=mistralai/Mistral-7B-Instruct-v0.2
MODEL_ID=Mistral-7B-Instruct-v0.2-dpo-Mistral-7B-Instruct-v0.2-branch_4-pref_pairs_${ROLE}

OUTPUT_DIR_ROOT=outputs # TODO outputs_MA

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
    --output_dir=${OUTPUT_DIR_ROOT}/${MODEL_ID} \
    --hub_model_id=${MODEL_ID} \
    # --prompt=qwen2-boxed
    # --do_eval \
    # --num_train_epochs=4 \
# python train.py configs/config_full.yaml \