#!/bin/bash

# Step 4: Value Iteration on Multi-Agent Trees
# This script performs value iteration to compute expected values for all nodes in the tree structure

set -e  # Exit on any error

# NAME=mistral-7b-instruct-v02-NESTED
NAME=mistral-7b-instruct-v02-branch_2
# Default paths (can be overridden with command line arguments)
INPUT_FILE="${1:-data_pipeline_MA/predictions/${NAME}-with-R.json}"
OUTPUT_FILE="${2:-data_pipeline_MA/predictions/${NAME}-with-V.json}"

echo "Step 4: Value Iteration on Multi-Agent Trees"
echo "============================================="
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run value iteration
echo "Running value iteration..."
python evaluation_MA/value_iteration_MA.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE"


echo ""
echo "Step 4 completed successfully!"
echo "Output saved to: $OUTPUT_FILE"
echo ""
echo "Usage: $0 [input_file] [output_file]"
echo "Example: $0 custom_input.json custom_output.json"
