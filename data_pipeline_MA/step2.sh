# NAME=mistral-7b-instruct-v02-FLAT
NAME=mistral-7b-instruct-v02-branch_2
export PRED_PATH=./data_pipeline_MA/predictions/${NAME}.json
export JUDGMENT_PATH=./data_pipeline_MA/predictions/${NAME}-judgment.json
export DATASET_PATH='./data/main_data.json'

HF_CACHE_DIR='./evaluation'

CUDA_VISIBLE_DEVICES=1 python3 evaluation_MA/evaluate_final_action_MA.py \
  --output-tree-format nested \
  --data-path $DATASET_PATH \
  --action-path $PRED_PATH \
  --step 'judge_leakage' \
  --output-path $JUDGMENT_PATH \
  --hf-cache-dir $HF_CACHE_DIR
#   --gpu-num 1 \
#   --model mistralai/Mistral-7B-Instruct-v0.2 \
