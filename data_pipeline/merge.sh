CORRECTIONS_PATH="data_pipeline/corrections/collect_corrections_output.json"
INPUT_DATA_PATH="data/main_data.json"
OUTPUT_PATH="data_pipeline/dataset.json"
PROMPT_TYPE="naive"

python data_pipeline/generate_dataset.py --corrections-path "$CORRECTIONS_PATH" --input-data-path "$INPUT_DATA_PATH" --output-path "$OUTPUT_PATH" --prompt-type "$PROMPT_TYPE"