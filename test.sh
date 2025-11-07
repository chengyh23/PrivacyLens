#!/bin/bash

# Switch from conda env step_dpo to privacylens:
# conda activate privacylens

# Single-anget
# PRED_MODEL=mistralai/Mistral-7B-Instruct-v0.2
# PRED_MODEL=Qwen/Qwen2.5-7B-Instruct
# PRED_MODEL=Qwen/Qwen2-0.5B-Instruct
# PRED_MODEL=outputs/qwen2-0.5b-instruct-step-dpo
PRED_MODEL=Qwen/Qwen2-72B-Instruct
# PRED_MODEL=outputs/qwen2-72b-instruct-step-dpo

EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# EVAL_STEP=judge_leakage
EVAL_STEP=helpfulness

# # Multi-agent
# MODEL_ID=qwen2-0.5b-instruct-dpo-refiner
# DATA_PATH=data_pipeline_MA/pref_pairs/pref_pairs_refiner.json


# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_cpu.yaml --mixed_precision bf16 \
#     --num_processes 2 \
CUDA_VISIBLE_DEVICES=6 python test.py \
    --input-path ./data/main_data_test.json \
    --num -1 \
    --prompt-type naive \
    --eval-step ${EVAL_STEP} \
    --pred-model ${PRED_MODEL} \
    --eval-model ${EVAL_MODEL} \
    --gpu-memory-utilization 0.56 \
    --gpu-num 1
    # --specific-case-name main83 \
    # --model_name_or_path="Qwen/Qwen2-72B-Instruct" \
    # --output_dir=outputs/qwen2-72b-instruct-step-dpo \
    # --hub_model_id=qwen2-72b-instruct-step-dpo \
# python train.py configs/config_full.yaml \