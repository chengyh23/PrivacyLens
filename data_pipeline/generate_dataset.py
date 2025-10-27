import argparse
import json
import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent

sys.path.append(str((script_dir / "../evaluation").resolve()))
sys.path.append(str((script_dir / "../data_construction").resolve()))
sys.path.append(str((script_dir / "../helper").resolve()))
from get_final_action import prepare_agent_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset from corrections and input data")
    parser.add_argument(
        "--corrections-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "corrections", "collect_corrections_output.json"),
        help="Path to corrections JSON file"
    )
    parser.add_argument(
        "--input-data-path",
        type=str,
        default="./data/main_data.json",
        help="Path to the main input data JSON"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset.json"),
        help="Output path for processed dataset JSON"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="naive",
        help="Prompt type to use in prepare_agent_prompt"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.corrections_path, "r") as f:
        corrections = json.load(f)

    with open(args.input_data_path, "r") as f:
        input_data = json.load(f)

    output_dataset = []
    for name, v in corrections.items():
        # Gather fields from corrections json
        chosen = v.get("chosen", "")
        rejected = v.get("rejected", "")

        # Parse the index from the name, which should be formatted like "mainX"
        if name.startswith("main"):
            try:
                idx = int(name[4:])
            except ValueError:
                raise ValueError(f"Invalid name format in corrections: {name}")
        else:
            raise ValueError(f"Invalid name format in corrections: {name}")
        # Defensive check for idx in data range
        if idx < 0 or idx >= len(input_data):
            raise IndexError(f"Index {idx} extracted from {name} is out of range for dataset of size {len(input_data)}")

        prompt = prepare_agent_prompt(
            prompt_type=args.prompt_type,
            user_name=input_data[idx]['trajectory']['user_name'],
            user_email=input_data[idx]['trajectory']['user_email'],
            user_instruction=input_data[idx]['trajectory']['user_instruction'],
            toolkits=input_data[idx]['trajectory']['toolkits'],
            executable_trajectory=input_data[idx]['trajectory']['executable_trajectory'],
            final_action=input_data[idx]['trajectory']['final_action']
        )
        
        output = {
            "id": name,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        output_dataset.append(output)
    # Save output as dataset.json
    with open(args.output_path, "w") as f:
        json.dump(output_dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset written to {args.output_path}")

if __name__ == "__main__":
    main()
