
# Multi-Agent Data Pipeline Script

START_STEP=4

MODEL=mistralai/Mistral-7B-Instruct-v0.2
MODEL_GENERATOR=${MODEL}
MODEL_VERIFIER=${MODEL}
MODEL_REFINER=${MODEL}
PRED_MODEL_ID=Mistral-7B-Instruct-v0.2

N_BRANCHING=4
EVAL_STEP='judge_leakage'   # 'helpfulness'
EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2


NAME=${PRED_MODEL_ID}-branch_${N_BRANCHING}
ACTION_PATH=./data_pipeline_MA/predictions/${NAME}.json
JUDGMENT_PATH=./data_pipeline_MA/predictions/${NAME}-judgment.json
JUDGMENT_VALUE_PATH="data_pipeline_MA/predictions/${NAME}-judgment-with-V.json"
PREF_DATA_PATH_VERIFIER="data_pipeline_MA/predictions/${NAME}-pref_pairs_verifier.json"
PREF_DATA_PATH_REFINER="data_pipeline_MA/predictions/${NAME}-pref_pairs_refiner.json"


DATASET_PATH='./data/main_data.json'
HF_CACHE_DIR='./evaluation'
# HF_CACHE_DIR='~/.cache/huggingface'

PROMPT_TYPE='naive' # or 'privacy_enhanced'
CUDA_VISIBLE_DEVICES=4


# ----- Step 1: Generate Multi-Agent Branching Tree -----
if [ $START_STEP -le 1 ]; then

    echo "Step 1: Generating multi-agent branching data..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python evaluation_MA/get_final_action_MA.py \
      --input-path $DATASET_PATH \
      --output-path $ACTION_PATH \
      --prompt-type $PROMPT_TYPE \
      --start-index 0 \
      --num-case -1 \
      --n $N_BRANCHING \
      --model-generator ${MODEL_GENERATOR} \
      --model-verifier ${MODEL_VERIFIER} \
      --model-refiner ${MODEL_REFINER} \
      --gpu-num 1 \
      --gpu-memory-utilization 0.5 \
      --output-tree-format nested
    echo "Step 1 completed."
    echo ""
fi

# ----- Step 2: Evaluate and Judge Leakage -----
if [ $START_STEP -le 2 ]; then

    echo "Step 2: Evaluating generated action tree..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python3 evaluation_MA/evaluate_final_action_MA.py \
      --output-tree-format nested \
      --data-path $DATASET_PATH \
      --action-path ${ACTION_PATH} \
      --step ${EVAL_STEP} \
      --model ${EVAL_MODEL} \
      --output-path ${JUDGMENT_PATH} \
      --gpu-memory-utilization 0.5 \
      --hf-cache-dir $HF_CACHE_DIR
    echo "Step 2 completed."
    echo ""
fi

# ----- Step 3: Reward Shaping -----
if [ $START_STEP -le 3 ]; then

    echo "Step 3: Reward assignment and Value iteration..."
    # mkdir -p "$(dirname "$OUTPUT_PATH")"
    python evaluation_MA/value_iteration_MA.py \
        --action-path "$ACTION_PATH" \
        --flat-judgment "$JUDGMENT_PATH" \
        --output-path "$JUDGMENT_VALUE_PATH"
    echo "Step 3 completed."
    echo ""
fi

# ----- Step 4: Generate Preference Pairs -----
if [ $START_STEP -le 4 ]; then

    echo "Step 4: Generating preference pairs for verifier and refiner..."

    python evaluation_MA/preference_pairs_MA.py \
        --input_path "$JUDGMENT_VALUE_PATH" \
        --output_path_V "$PREF_DATA_PATH_VERIFIER" \
        --output_path_R "$PREF_DATA_PATH_REFINER"

    if [ $? -eq 0 ]; then
        echo "Step 4 completed."
        echo ""
    else
        echo "Step 4: Error: Failed to generate preference pairs."
        exit 1
    fi
fi


