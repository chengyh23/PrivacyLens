#!/bin/bash

# TEST_DATA_PATH=./data/main_data_test.json
TEST_DATA_PATH=data_pipeline_MA/predictions/Mistral-7B-Instruct-v0.2-branch_4-pref_pairs_empty_cases.txt


MODEL_GENERATOR=mistralai/Mistral-7B-Instruct-v0.2
MODEL_PREFIX=outputs/Mistral-7B-Instruct-v0.2-dpo-Mistral-7B-Instruct-v0.2-branch_4-pref_pairs
MODEL_VERIFIER=${MODEL_PREFIX}_verifier
MODEL_REFINER=${MODEL_PREFIX}_refiner

EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2
EVAL_STEP=("judge_leakage" "helpfulness")


# for PRED_MODEL in "${PRED_MODELS[@]}"; do
#     CUDA_VISIBLE_DEVICES=2 python test.py \
#         --input-path ${TEST_DATA_PATH} \
#         --num -1 \
#         --prompt-type naive \
#         --eval-step "${EVAL_STEP[@]}" \
#         --pred-model ${PRED_MODEL} \
#         --eval-model ${EVAL_MODEL} \
#         --gpu-memory-utilization 0.56 \
#         --gpu-num 1
#         # --specific-case-name main83 \
#         # --model_name_or_path="Qwen/Qwen2-72B-Instruct" \
#         # --output_dir=outputs/qwen2-72b-instruct-step-dpo \
#         # --hub_model_id=qwen2-72b-instruct-step-dpo \
# done

CUDA_VISIBLE_DEVICES=2 python test_MA.py \
    --input-path ${TEST_DATA_PATH} \
    --num -1 \
    --prompt-type naive \
    --eval-step "${EVAL_STEP[@]}" \
    --model-generator ${MODEL_GENERATOR} \
    --model-verifier ${MODEL_VERIFIER} \
    --model-refiner ${MODEL_REFINER} \
    --eval-model ${EVAL_MODEL} \
    --gpu-memory-utilization 0.3 \
    --gpu-num 1 \
    # --output-tree-format nested