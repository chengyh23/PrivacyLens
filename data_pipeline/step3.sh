# # Analyze the leakage


# # export OPENAI_BASE_URL="https://litellm.oit.duke.edu/v1" # input openai base_url here
# # export OPENAI_API_KEY="sk-mIfo3TvnCCeYnwG0djaj5A" # input openai api_key here

# export OPENAI_BASE_URL=http://hl279-cmp-01.egr.duke.edu:4000/v1
# export OPENAI_API_KEY=sk-lEkSb3ptAWNCvlS2nepzLA

# python3 data_pipeline/locate_leak_by_gpt4.py \
#     --prompt "qwen2-boxed-step" \
#     --save_dir "./data_pipeline/generated" \
#     --json_files "./data_pipeline/predictions/qwen2-7b-instr*.csv" \
#     --max_count_total 100



# # Collect leaks from predictions and judgements

# PREFIX="qwen2-7b-instr"
PREFIX="qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group"
PREDICTIONS_PATH="data_pipeline/predictions/${PREFIX}0.csv"
JUDGEMENTS_PATH="data_pipeline/predictions/${PREFIX}0.json"
OUTPUT_PATH_JSON="data_pipeline/leaks/${PREFIX}0.json"

python3 data_pipeline/collect_leaks.py --predictions $PREDICTIONS_PATH --judgements $JUDGEMENTS_PATH --leaks_json $OUTPUT_PATH_JSON

