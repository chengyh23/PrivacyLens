# conda activate privacylens
START_STEP=3

# PRED_MODEL="Qwen/Qwen3-4B-Instruct-2507"
# PRED_MODEL="Qwen/Qwen2.5-7B-Instruct"
# PRED_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# PRED_MODEL="meta-llama/Llama-3.1-8B-Instruct"
PRED_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
# PRED_MODEL="google/gemma-3-1b-it"
# PRED_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# PRED_MODEL=claude-4-sonnet-20250514
# PRED_MODEL=gpt-5
PRED_MODEL_ID=Mistral-7B-Instruct-v0.3

N_SAMPLE_PER_CASE=10
EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2
EVAL_STEP='judge_leakage'   # 'helpfulness'

ACTION_PATH=data_pipeline_aug/pref_pairs_augmented/train_aug-${PRED_MODEL_ID}-${N_SAMPLE_PER_CASE}.json
JUDGE_PATH=data_pipeline_aug/pref_pairs_augmented/train_aug_eval-${PRED_MODEL_ID}-${N_SAMPLE_PER_CASE}.json
PREF_DATA_PATH=data_pipeline_aug/pref_pairs_augmented/preference_pairs-${PRED_MODEL_ID}-${N_SAMPLE_PER_CASE}.json
# data_pipeline/pref_pairs_augmented/train_aug-Meta-Llama-3-8B-Instruct.json

# INPUT="data_pipeline/pref_pairs/train.json"
DATASET_PATH='./data/main_data.json'
HF_CACHE_DIR='./evaluation'
PROMPT_TYPE='naive' # or 'privacy_enhanced'
CUDA_VISIBLE_DEVICES=1

if [ $START_STEP -le 1 ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python data_pipeline_aug/sample_final_action.py \
        --input-path $DATASET_PATH \
        --output-path $ACTION_PATH \
        --start-index 0 \
        --num-case -1 \
        --n-sample-per-case ${N_SAMPLE_PER_CASE} \
        --pred-model ${PRED_MODEL} \
        --prompt-type $PROMPT_TYPE \
        --gpu-memory-utilization 0.5
fi


if [ $START_STEP -le 2 ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python data_pipeline_aug/eval_final_action.py \
        --data-path $DATASET_PATH \
        --action-path $ACTION_PATH \
        --output-path ${JUDGE_PATH} \
        --eval-step $EVAL_STEP \
        --eval-model ${EVAL_MODEL}\
        --gpu-memory-utilization 0.3 \
        --hf-cache-dir $HF_CACHE_DIR
        # --step 'judge_leakage' \
fi

if [ $START_STEP -le 3 ]; then
    python data_pipeline_aug/gen_preference_pairs.py \
        --eval-data-path ${JUDGE_PATH} \
        --pref-data-path ${PREF_DATA_PATH} \
        --verbose
fi
