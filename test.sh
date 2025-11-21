#!/bin/bash

# Switch from conda env step_dpo to privacylens:
# conda activate privacylens

# Single-anget

# TEST_DATA_PATH=./data/main_data_test.json
TEST_DATA_PATH=data_pipeline/pref_pairs_augmented/preference_pairs-Mistral-7B-Instruct-v0.3-10_empty_cases.txt

PRED_MODELS=(
    outputs/Mistral-7B-Instruct-v0.3-dpo-preference_pairs-Mistral-7B-Instruct-v0.3-10
    mistralai/Mistral-7B-Instruct-v0.3
    # outputs/Meta-Llama-3-8B-Instruct-dpo-preference_pairs-Meta-Llama-3-8B-Instruct-10
    # meta-llama/Meta-Llama-3-8B-Instruct
)

EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2
EVAL_STEP=("judge_leakage" "helpfulness")


for PRED_MODEL in "${PRED_MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python test.py \
        --input-path ${TEST_DATA_PATH} \
        --num -1 \
        --prompt-type naive \
        --eval-step "${EVAL_STEP[@]}" \
        --pred-model ${PRED_MODEL} \
        --eval-model ${EVAL_MODEL} \
        --gpu-memory-utilization 0.56 \
        --gpu-num 1
        # --specific-case-name main83 \
        # --model_name_or_path="Qwen/Qwen2-72B-Instruct" \
        # --output_dir=outputs/qwen2-72b-instruct-step-dpo \
        # --hub_model_id=qwen2-72b-instruct-step-dpo \
done