export PRED_PATH='./data_pipeline/predictions/Mistral-7B-Instruct-v0.2'
# export PRED_PATH='./data_pipeline/predictions/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'
# export JUDGELEAK_PATH='./data_pipeline/judgements/qwen2-7b-instr'
# export JUDGELEAK_PATH='./data_pipeline/judgements/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'

export DATASET_PATH='./data/main_data.json'

MODEL_NAME_OR_PATH=mistralai/Mistral-7B-Instruct-v0.2
HF_CACHE_DIR='./evaluation'

CUDA_VISIBLE_DEVICES=1 \
python evaluation/evaluate_final_action.py \
    --data-path $DATASET_PATH \
    --step 'judge_leakage' \
    --action-path "${PRED_PATH}.csv" \
    --output-path "${PRED_PATH}.json" \
    --model ${MODEL_NAME_OR_PATH}\
    --gpu-memory-utilization 0.5 \
    --hf-cache-dir $HF_CACHE_DIR
    # --step 'helpfulness' \