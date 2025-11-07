import argparse
import json
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

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



def get_ids_from_split(split_name: str, pref_pairs_dir: str = "data_pipeline/pref_pairs"):
    """
    Loads all prompt ids (e.g., main123) from split files in data_pipeline/pref_pairs/{split_name}.jsonl
    Expects that each line in the file has a field: "id": "mainXXX"
    """
    ids = set()
    
    split_file = os.path.join(pref_pairs_dir, f"{split_name}.json")
    if not os.path.exists(split_file):
        raise RuntimeError(f"Pref pairs split file does not exist: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            id_value = item.get("id")
            if id_value is None:
                raise ValueError(f'Missing "id" field in pref_pairs split file: {split_file}')
            ids.add(id_value)
    return ids

def split_dataset_by_pref_pairs(dataset_path: str):
    """
    Splits the output_dataset (list of dicts) into train/val/test splits using the ids in pref_pairs split files.
    Returns: {split_name: [dict,...]}
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    split_names = ["train", "validation", "test"]
    split_ids = {split: get_ids_from_split(split) for split in split_names}
    id_to_item = {item["name"]: item for item in dataset}

    splits = {split: [] for split in split_names}
    for split in split_names:
        for id_ in split_ids[split]:
            if not id_ in id_to_item:
                error_msg = f"ID {id_} not found in output_dataset"
                raise ValueError(error_msg)
            splits[split].append(id_to_item[id_])
    
    for split in split_names:
        out_filename = dataset_path.replace(".json", f"_{split}.json")
        if os.path.exists(out_filename):
            print(f"{out_filename} already exists, skipping.")
            continue
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(splits[split], f, indent=2, ensure_ascii=False)
    return splits


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
    
    # # Save output as dataset.json
    # with open(args.output_path, "w") as f:
    #     json.dump(output_dataset, f, indent=2, ensure_ascii=False)
    # print(f"Dataset written to {args.output_path}")
    
    # Split the dataset and save in separate files: train (80%), validation (10%), test (10%)
    train_data, tmp_data = train_test_split(output_dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(tmp_data, test_size=0.5, random_state=42)

    base = args.output_path.rsplit(".", 1)[0]
    train_path = base + "_train.json"
    val_path = base + "_val.json"
    test_path = base + "_test.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Splits written to {train_path}, {val_path}, {test_path}")

if __name__ == "__main__":
    # main()
    split_dataset_by_pref_pairs(dataset_path = "data/main_data.json")