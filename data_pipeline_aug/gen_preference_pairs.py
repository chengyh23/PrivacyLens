import os
import argparse
import json
from collections import defaultdict
from typing import List

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def cartesian_product(pos_set: List[str], neg_set: List[str], label: str, prompt: str = None):
    r"""
    Return:
        prod (List[dict]):
            "name": str
            "chosen": str
            "rejected": str
    """
    prod = []
    for e1 in pos_set:
        for e2 in neg_set:
            pair = {
                "name": label,
                "chosen": e1,
                "rejected": e2,
                **({"prompt": prompt} if prompt is not None else {})
            }
            prod.append(pair)
    return prod
    
def main(eval_data_path: str, pref_data_path: str, verbose: bool = False):

    data = load_json(eval_data_path)

    # Iterate over the data to get start and end indices for each 'name' (mainX)
    name_indices = defaultdict(list)
    for idx, entry in enumerate(data):
        name = entry["name"]
        name_indices[name].append(idx)
    # Generate preference pairs and test cases respectively
    pref_pairs = []
    test_case_names = []
    for name, indices in name_indices.items():
        pos_actions, neg_actions = [], []
        for index in indices:
            entry = data[index]
            assert name == entry["name"]
            
            if entry["eval_result"]["leak_info"] is True:
                neg_actions.append(entry["final_action"])
            elif entry["eval_result"]["leak_info"] is False:
                pos_actions.append(entry["final_action"])
        case_pref_pairs = cartesian_product(pos_actions, neg_actions, name)
        
        if len(case_pref_pairs)==0:
            test_case_names.append(name)
        if verbose:
            print(f'{name}: {len(pos_actions):<2} x {len(neg_actions):<2} = {len(case_pref_pairs):<4}')
        pref_pairs += case_pref_pairs
    print(f'Total number of preference pairs: {len(pref_pairs)}')

    # Save preference pairs
    if os.path.exists(pref_data_path):
        with open(pref_data_path, "r", encoding="utf-8") as fin:
            pref_pairs_existing = json.load(fin)
        print(f"Already exists: {pref_data_path}\n(Total # preference pairs: {len(pref_pairs_existing)})")
    else:
        with open(pref_data_path, "w", encoding="utf-8") as fout:
            json.dump(pref_pairs, fout, indent=2, ensure_ascii=False)
    
    # Save test cases
    test_case_path = os.path.splitext(pref_data_path)[0] + "_empty_cases.txt"
    if os.path.exists(test_case_path):
        print(f"Already exists: {test_case_path}\n(Total # test cases: {len(test_case_names)})")
    else:
        with open(test_case_path, "w", encoding="utf-8") as fout:
            for name in test_case_names:
                fout.write(f"{name}\n")
        print(f"Wrote out test cases' names to {test_case_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate preference pairs from evaluation data."
    )
    parser.add_argument(
        "--eval-data-path",
        required=True,
        help="Path to the evaluation data JSON file.",
    )
    parser.add_argument(
        "--pref-data-path",
        default="data_pipeline/pref_pairs_augmented/preference_pairs-Mistral-7B-Instruct-v0.3-10.json",
        help="Destination path for the generated preference pairs JSON file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generation statistics for each case.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.eval_data_path, args.pref_data_path, args.verbose)
