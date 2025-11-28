""" Johan C
Nov, 2025


"""
import logging
import argparse
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from evaluation.get_final_action import get_final_actions
from evaluation.evaluate_final_action import evaluate_final_actions, calc_eval_metrics
from huggingface_hub import model_info

logger = logging.getLogger(__name__)

def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist!")
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        # fallback: maybe files saved directly to output_dir
        #           e.g., outpur_dir=outputs_test/Meta-Llama-3-8B-Instruct/checkpoint-800
        return output_dir
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(output_dir, checkpoints[-1])

# # TODO to be deleted
# def calc_leak_rate(eval_output_path):
#     with open(eval_output_path, "r") as f:
#         results = json.load(f)
#     # Format: {name: {"leak_info": bool, ...}, ...}
#     n_total = len(results)
#     n_leak = sum(1 for v in results.values() if (isinstance(v, dict) and v.get("leak_info", False)))
#     leak_rate = n_leak / n_total if n_total > 0 else 0.0
#     print(f"Leakage rate: {n_leak}/{n_total} = {leak_rate:.3f}")

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path of the TEST data in json/txt format.')
    # parser.add_argument('--output-path', type=str, required=True,
    #                     help='Path to save the probing results in csv format.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the cases to evaluate.')
    parser.add_argument('--num', type=int, default=1,
                        help='Number of cases to evaluate. If -1, evaluate all remaining cases.')
    parser.add_argument('--specific-case-name', type=str, default=None,
                        help='If not None, only evaluate the case with the given name.')
    parser.add_argument('--prompt-type', type=str,
                        choices=['naive', 'privacy_enhanced', 'conservative', 'reckless'],
                        help='The type of the prompt to use for the agent.')
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs to use for VLLM.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.')
    parser.add_argument('--hf-cache-dir', type=str,
                        help='The cache directory for the Hugging Face model.')
    parser.add_argument('--pred-model', type=str,
                        help='The model to use for generating action.')
    parser.add_argument("--checkpoint_dir", type=str,  
                        help="Path to model outputs/checkpoints")
    parser.add_argument('--eval-model', type=str,
                        help='The model to use for evaluating action.')
    parser.add_argument("--eval_result", type=str, default=None, help="Path to eval leakage json file (if already evaluated)")
    parser.add_argument("--eval_final_action_path", type=str, default="evaluation/evaluate_final_action.py", help="Evaluation script")
    parser.add_argument("--eval-step", type=str, nargs='+',
                        choices=['extract_secret', 'judge_leakage', 'helpfulness'])
    parser.add_argument("--pref_data_path", type=str, default="data_pipeline/pref_pairs", help="Eval data path (dir or file)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation (unused if not re-running eval)")
    return parser.parse_args()


def main():
    args = prepare_args()
    load_dotenv()
    
    log_level = logging.INFO
    logger.setLevel(log_level)

    # If eval_model looks like a path with a '/', replace it with outputs_test/ version
    if args.pred_model is None:
        raise ValueError("You must specify --pred-model.")

    if not os.path.exists(args.pred_model):
        pred_model_name_or_path = args.pred_model
        outputs_test_dir = "outputs_test/" + args.pred_model.split("/", 1)[-1]
    else:
        pred_model_name_or_path = find_latest_checkpoint(args.pred_model)
        outputs_test_dir = pred_model_name_or_path.replace("outputs", "outputs_test")    
    print(f"Pred using: {pred_model_name_or_path}, output to {outputs_test_dir}")
    print(f"Eval using: {args.eval_model}")
    
    # info = model_info(args.checkpoint_dir)
    # print(info.card_data.get("base_model"))
    
    os.makedirs(outputs_test_dir, exist_ok=True)
    output_dir = os.path.join(outputs_test_dir, "predictions")
    
    #########################
    # Get final action
    #########################
    output_path_actions = os.path.join(output_dir, "actions.csv")
    if not os.path.exists(output_path_actions):
        result = get_final_actions(args, model_name_or_path=pred_model_name_or_path, num_case=args.num)
        
        os.makedirs(output_dir, exist_ok=True)
        try:
            pd.DataFrame(result).to_csv(output_path_actions, index=False)
            print(f"Successfully wrote actions to {output_path_actions}")
        except Exception as e:
            print(f'Error: {e}')
            with open(output_path_actions.replace('.csv', 'json'), 'w') as f:
                json.dump(result, f)
    else:
        logger.info(f"{output_path_actions} exists, skipping get_final_actions and loading existing results.")
        print(f"{output_path_actions} exists, skipping get_final_actions and loading existing results.")
    
    #########################
    # Evaluate final action
    #########################
    output_path_eval = os.path.join(output_dir, f"evaluations.json")
    # output_path_eval = os.path.join(output_dir, f"{args.eval_step}.csv")
    if not os.path.exists(output_path_eval):
        name_to_result = evaluate_final_actions(
            args, 
            model_name_or_path=args.eval_model, 
            data_path=args.input_path, 
            actions=output_path_actions,
            step=args.eval_step
        )
        with open(output_path_eval, 'w') as f:
            json.dump(name_to_result, f, indent=4)
    else:
        print(f"{output_path_eval} exists, skipping evaluate_final_actions and loading existing results.")
    
    calc_eval_metrics(output_path_eval, args.eval_step, verbose=False)

if __name__ == "__main__":
    # main()

    # calc_leak_rates("data_pipeline/pref_pairs_augmented/train_aug_eval-Mistral-7B-Instruct-v0.3-ALL.json")
    # summarize_leak_rates("data_pipeline/pref_pairs_augmented")
    # evals = ["outputs_test/Meta-Llama-3-8B-Instruct/predictions/evaluations.csv", "outputs_test/Meta-Llama-3-8B-Instruct-dpo-preference_pairs-Meta-Llama-3-8B-Instruct-10/checkpoint-1000/predictions/evaluations.csv"]
    # evals = ["outputs_test/Mistral-7B-Instruct-v0.3/predictions/evaluations.csv", "outputs_test/Mistral-7B-Instruct-v0.3-dpo-preference_pairs-Mistral-7B-Instruct-v0.3-10/checkpoint-1400/predictions/evaluations.csv"]
    # calc_eval_metrics(evals)
    
    from evaluation.evaluate_final_action import check_action_validity_for_predictions
    for action_path in [
        "data_pipeline_aug/pref_pairs_augmented/train_aug-Mistral-7B-Instruct-v0.3-10.json",
        "data_pipeline_aug/pref_pairs_augmented/train_aug-Meta-Llama-3-8B-Instruct-10.json"]:
        check_action_validity_for_predictions(action_path)