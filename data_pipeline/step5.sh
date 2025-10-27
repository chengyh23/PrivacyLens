# export PRED_2_PATH='./data_pipeline/corrections/qwen2-7b-instr'
export PRED_2_PATH='./data_pipeline/predictions2/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'
# export JUDGELEAK_PATH='./data_pipeline/judgements/qwen2-7b-instr'
# export JUDGELEAK_PATH='./data_pipeline/judgements/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'

export DATASET_PATH='./data/main_data.json'

HF_CACHE_DIR='./evaluation'
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_final_action.py --data-path $DATASET_PATH --action-path $PRED_2_PATH"0.csv" --step 'judge_leakage' --output-path $PRED_2_PATH"0.json" --hf-cache-dir $HF_CACHE_DIR

