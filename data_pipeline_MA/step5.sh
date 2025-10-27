#!/bin/bash

echo "Step 5: Generating preference pairs for verifier and refiner..."

# Set the paths
INPUT_PATH="data_pipeline_MA/predictions/mistral-7b-instruct-v02-NESTED-with-V.json"
OUTPUT_PATH_V="data_pipeline_MA/pref_pairs_verifier.json"
OUTPUT_PATH_R="data_pipeline_MA/pref_pairs_refiner.json"

# Check if input file exists
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file '$INPUT_PATH' does not exist."
    echo "Please run previous steps to generate the nested tree data first."
    exit 1
fi

# Run the preference pairs generation script with parameters
python evaluation_MA/preference_pairs_MA.py \
    --input_path "$INPUT_PATH" \
    --output_path_V "$OUTPUT_PATH_V" \
    --output_path_R "$OUTPUT_PATH_R"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Successfully generated preference pairs:"
    echo "  - Verifier pairs: $OUTPUT_PATH_V"
    echo "  - Refiner pairs: $OUTPUT_PATH_R"
else
    echo "Error: Failed to generate preference pairs."
    exit 1
fi

echo "Step 5 completed successfully!"
