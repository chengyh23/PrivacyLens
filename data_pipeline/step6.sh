LEAKS_JSON_PATH="data_pipeline/leaks/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group0.json"
PRED_2_PATH="data_pipeline/predictions2/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group0.json"
JUDGMENT_2_PATH="data_pipeline/predictions2/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group0.csv"
OUTPUT_PATH="data_pipeline/corrections/collect_corrections_output.json"

python data_pipeline/collect_corrections.py \
  --leaks-json-path="$LEAKS_JSON_PATH" \
  --pred-json-path="$PRED_2_PATH" \
  --pred-csv-path="$JUDGMENT_2_PATH" \
  --output-path="$OUTPUT_PATH"