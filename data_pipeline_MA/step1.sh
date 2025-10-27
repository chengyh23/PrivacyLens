
export MODEL_GENERATOR='mistralai/Mistral-7B-Instruct-v0.2'
export MODEL_VERIFIER='mistralai/Mistral-7B-Instruct-v0.2'
export MODEL_REFINER='mistralai/Mistral-7B-Instruct-v0.2'
# export MODEL_GENERATOR='meta-llama/Meta-Llama-3-8B-Instruct'
# export MODEL_VERIFIER='meta-llama/Meta-Llama-3-8B-Instruct'
# export MODEL_REFINER='meta-llama/Meta-Llama-3-8B-Instruct'

N_BRANCHING=2

export OUTPUT_PATH="./data_pipeline_MA/predictions/mistral-7b-instruct-v02-branch_${N_BRANCHING}.json"
export DATASET_PATH='./data/main_data.json'
export HF_CACHE_DIR='~/.cache/huggingface'

PROMPT_TYPE='naive' # or 'privacy_enhanced'


CUDA_VISIBLE_DEVICES=1 python evaluation_MA/get_final_action_MA.py \
  --input-path $DATASET_PATH \
  --output-path $OUTPUT_PATH \
   --prompt-type $PROMPT_TYPE \
  --start-index 0 \
  --num 1 \
  --n $N_BRANCHING \
  --model-generator $MODEL_GENERATOR \
  --model-verifier $MODEL_VERIFIER \
  --model-refiner $MODEL_REFINER \
  --gpu-num 1 \
  --output-tree-format nested &

