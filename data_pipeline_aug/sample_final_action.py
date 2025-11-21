import argparse
import json
import random
from tqdm import tqdm
from openai import OpenAI
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from evaluation.get_final_action import get_final_actions_aug, get_final_actions

client = OpenAI()
def generate(prompt: str, n: int) -> str:
    generations = []
    for i in range(n):
        resp = client.responses.create(
            model="gpt-5",
            input=prompt,
            # seed=random.randint(1, 1_000_000_000),  # new random seed each call

        )
        generations.append(resp.output_text.strip()) # Convenient helper to get the text
    return generations

def print_action_counts_by_id(json_path):
    """
    Reads the data at `json_path` and prints a count of actions/entries for each unique id ("mainxx").
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_counts = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        _id = entry.get('id', None)
        if _id is not None:
            id_counts[_id] = id_counts.get(_id, 0) + 1
    for _id, count in sorted(id_counts.items()):
        print(f"id: {_id}  count: {count}")

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path of the data in json format.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the probing results in csv format.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the cases to evaluate.')
    parser.add_argument('--num-case', type=int, default=1,
                        help='Number of cases to evaluate. If -1, evaluate all remaining cases.')
    parser.add_argument('--specific-case-name', type=str, default=None,
                        help='If not None, only evaluate the case with the given name.')
    parser.add_argument('--prompt-type', type=str,
                        choices=['naive', 'privacy_enhanced', 'conservative', 'reckless'],
                        help='The type of the prompt to use for the agent.')
    parser.add_argument('--pred-model', type=str, required=True,
                        help='The model to use for generating action.')
    parser.add_argument('--n-sample-per-case', type=int, required=True,
                        help='N generations per case.')
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs to use for VLLM.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.')
    parser.add_argument('--hf-cache-dir', type=str,
                        help='The cache directory for the Hugging Face model.')

    return parser.parse_args()

if __name__ == "__main__":
    # print_action_counts_by_id("data_pipeline/pref_pairs_augmented/train_aug.json")
    
    args = prepare_args()

    start = args.start_index    
    
    if not os.path.exists(args.output_path):
        generations = get_final_actions(
        # generations = get_final_actions_aug(
            args, 
            model_name_or_path=args.pred_model, 
            num_case=args.num_case,
            n_sample_per_case=args.n_sample_per_case
        )
        # print(generations)    # debug


        # Store id and generations for each case
        names, final_actions = generations['name'], generations['final_action']
        aug_cases = []
        for name, action in zip(names, final_actions):
            aug_cases.append({
                "id": name,
                "pred_model": args.pred_model,
                "final_action": action
            })

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(aug_cases, f, indent=2, ensure_ascii=False)
            print(f"Augmented samples written to {args.output_path} ({len(aug_cases)} cases).")
    else:
        print(f"{args.output_path} exists, skipping get_final_actions_aug and loading existing results.")
