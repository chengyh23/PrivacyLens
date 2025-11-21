import os
import argparse
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from evaluation.evaluate_final_action import evaluate_final_actions, calc_leak_rates, calc_avg_helpfulness

def summarize_eval_metrics(eval_datas):
    """
    eval_datas:
        List[str]: list of paths to the evaluation data
        str: path to the directory containing the evaluation data
    """
    if isinstance(eval_datas, str):
        eval_data_paths = [f for f in os.listdir(eval_datas) if f.startswith("train_aug_eval") and f.endswith(".json")]
        print(eval_data_paths)
        eval_data_paths = [os.path.join(eval_datas, f) for f in eval_data_paths]
    else:
        eval_data_paths = eval_datas

    for eval_data_path in eval_data_paths:
        calc_leak_rates(eval_data_path, verbose=False)
        calc_avg_helpfulness(eval_data_path, verbose=False)

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path of the evaluation data in json format.')
    parser.add_argument('--action-path', type=str, help='Path of the LM agent final actions.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the results.')
    parser.add_argument('--eval-step', type=str,
                        choices=['extract_secret', 'judge_leakage', 'helpfulness'])
    parser.add_argument('--eval-model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs to use.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.')
    parser.add_argument('--hf-cache-dir', type=str,
                        help='The cache directory for the Hugging Face model.')
    return parser.parse_args()

def main():
    # Read from train_aug.json and return list of dict {name: id, final_action: "Thought ..."}
    args = prepare_args()

    # Read actions
    with open(args.action_path, "r") as f:
        actions_json = json.load(f)
    actions = []
    for item in actions_json:
        action_entry = dict(item)
        action_entry["name"] = action_entry.pop("id")
        actions.append(action_entry)

    if not os.path.exists(args.output_path):
        name_and_result = evaluate_final_actions(
            args, 
            model_name_or_path=args.eval_model, 
            data_path=args.data_path, 
            actions=actions,
            step=args.eval_step
        )
        with open(args.output_path, 'w') as f:
            json.dump(name_and_result, f, indent=4)
        print("Evaluation complete. Results written to", args.output_path)
    else:
        print(f"{args.output_path} exists, skipping evaluate_final_actions and loading existing results.")
    calc_leak_rates(args.output_path)

if __name__ == "__main__":
    # main()

    # calc_leak_rates("data_pipeline/pref_pairs_augmented/train_aug_eval-Mistral-7B-Instruct-v0.3-ALL.json")
    # summarize_leak_rates("data_pipeline/pref_pairs_augmented")
    # evals = ["outputs_test/Meta-Llama-3-8B-Instruct/predictions/evaluations.csv", "outputs_test/Meta-Llama-3-8B-Instruct-dpo-preference_pairs-Meta-Llama-3-8B-Instruct-10/checkpoint-1000/predictions/evaluations.csv"]
    evals = ["outputs_test/Mistral-7B-Instruct-v0.3/predictions/evaluations.csv", "outputs_test/Mistral-7B-Instruct-v0.3-dpo-preference_pairs-Mistral-7B-Instruct-v0.3-10/checkpoint-1400/predictions/evaluations.csv"]
    summarize_eval_metrics(evals)
