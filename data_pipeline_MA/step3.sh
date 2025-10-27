#!/bin/bash

# Step 3: Add reward values to nested tree structure
# This script takes the nested tree structure and adds reward values from flat judgments

set -e  # Exit on any error

# NAME=mistral-7b-instruct-v02-NESTED
NAME=mistral-7b-instruct-v02-branch_2
# Default paths (can be overridden with command line arguments)
ACTION_PATH="${1:-data_pipeline_MA/predictions/${NAME}.json}"
FLAT_JUDGMENT_FILE="${2:-data_pipeline_MA/predictions/${NAME}-judgment.json}"
OUTPUT_PATH="${3:-data_pipeline_MA/predictions/${NAME}-with-R.json}"

echo "Step 3: Adding reward values to nested tree structure"
echo "=================================================="
echo "Action (Nested) file: $ACTION_PATH"
echo "Flat judgment file: $FLAT_JUDGMENT_FILE"
echo "Output Tree (Nested) file with Reward: $OUTPUT_PATH"
echo ""


# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Run the reward shaping script
echo "Running reward shaping..."
python evaluation_MA/reward_shaping.py \
    --action-path "$ACTION_PATH" \
    --flat-judgment "$FLAT_JUDGMENT_FILE" \
    --output-path "$OUTPUT_PATH"

echo ""
echo "Step 3 completed successfully!"
echo "Output saved to: $OUTPUT_PATH"
