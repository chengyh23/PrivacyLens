START_STEP=1

PRED_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
PRED_MODEL_ID=Mistral-7B-Instruct-v0.3
EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2

N_SAMPLE_PER_CASE=10
DATASET_PATH='./data/main_data_train.json'
JUDGE_PATH=data_pipeline_aug/pref_pairs_iterative/eval-${PRED_MODEL_ID}-${N_SAMPLE_PER_CASE}.json
PREF_DATA_PATH=data_pipeline_aug/pref_pairs_iterative/preference_pairs-${PRED_MODEL_ID}-${N_SAMPLE_PER_CASE}.json

PROMPT_TYPE='naive' # or 'privacy_enhanced'
EVAL_STEP='judge_leakage'   # 'helpfulness'

CUDA_VISIBLE_DEVICES=2


if [ $START_STEP -le 1 ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python data_pipeline_aug/iterative_sample_and_eval.py \
        --input-path ${DATASET_PATH} \
        --output-path ${JUDGE_PATH} \
        --pred-model ${PRED_MODEL} \
        --eval-model ${EVAL_MODEL} \
        --eval-step ${EVAL_STEP} \
        --start-index 0 \
        --num-case -1 \
        --prompt-type ${PROMPT_TYPE} \
        --gpu-memory-utilization 0.5 \
        --n-sample-per-case ${N_SAMPLE_PER_CASE}
        # --specific-case-name main4 main5 \
fi

# JUDGE_PATH_MERGE=(
#     data_pipeline_aug/pref_pairs_iterative/eval-Mistral-7B-Instruct-v0.3-10-round0.json
#     data_pipeline_aug/pref_pairs_iterative/eval-Mistral-7B-Instruct-v0.3-10-round0.easy.json
# )
# if [ $START_STEP -le 2 ]; then
#     python data_pipeline_aug/gen_preference_pairs.py \
#         --eval-data-path "${JUDGE_PATH_MERGE[@]}" \
#         --pref-data-path ${PREF_DATA_PATH}
# fi