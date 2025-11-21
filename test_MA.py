""" Johan C
Nov, 2025


"""
import logging
import argparse
from tqdm import tqdm
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from evaluation.get_final_action import get_final_actions
from evaluation.evaluate_final_action import evaluate_final_actions, calc_eval_metrics
from evaluation_MA.get_final_action_MA import collect_trajectories
from huggingface_hub import model_info

logger = logging.getLogger(__name__)

from test import find_latest_checkpoint
from evaluation_MA.tree_utils import Forest
from evaluation.evaluate_final_action import evaluate_final_actions_batch



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
    parser.add_argument('--model-generator', type=str, required=True,
                        help='The model to use for generating action of Generator in the multi-agent system.')
    parser.add_argument('--model-verifier', type=str, required=True,
                        help='The model to use for generating action of Verifier in the multi-agent system.')
    parser.add_argument('--model-refiner', type=str, required=True,
                        help='The model to use for generating action of Refiner in the multi-agent system.')
    parser.add_argument("--checkpoint_dir", type=str,  
                        help="Path to model outputs/checkpoints")
    parser.add_argument('--eval-model', type=str, required=True,
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

    # TODO how to determine outputs_test_dir?
    if not os.path.exists(args.model_generator):
        model_name_or_path_G = args.model_generator
    else:
        model_name_or_path_G = find_latest_checkpoint(args.model_generator)

    if not os.path.exists(args.model_verifier):
        model_name_or_path_V = args.model_verifier
    else:
        model_name_or_path_V = find_latest_checkpoint(args.model_verifier)

    if not os.path.exists(args.model_refiner):
        model_name_or_path_R = args.model_refiner
        outputs_test_dir = "outputs_test/" + args.model_refiner.split("/", 1)[-1]
    else:
        model_name_or_path_R = find_latest_checkpoint(args.model_refiner)
        outputs_test_dir = model_name_or_path_R.replace("outputs", "outputs_test")
    

    print(f"Pred using: G: {model_name_or_path_G}, V: {model_name_or_path_V}, R: {model_name_or_path_R}. output to {outputs_test_dir}")
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
        result = collect_trajectories(
            args, 
            model_generator=model_name_or_path_G,
            model_verifier=model_name_or_path_V,
            model_refiner=model_name_or_path_R,
            num_case=args.num, 
            n_branching=1
        )
        
        os.makedirs(output_dir, exist_ok=True)
        result.to_dict(output_path_actions)
        print(f"Successfully wrote actions to {output_path_actions}")
    else:
        logger.info(f"{output_path_actions} exists, skipping get_final_actions and loading existing results.")
        print(f"{output_path_actions} exists, skipping get_final_actions and loading existing results.")
    
    #########################
    # Evaluate final action
    #########################
    output_path_eval = os.path.join(output_dir, f"evaluations.json")
    # output_path_eval = os.path.join(output_dir, f"{args.eval_step}.csv")
    if not os.path.exists(output_path_eval):
        forest = Forest.from_dict(output_path_actions)
        actions = []
        for i_case in tqdm(range(len(forest)), desc="loading actions of all cases", leave=False):
        # for i_case in tqdm(range(len(data))):
            # assert forest[i_case].name == f'main{i_case+1}', f"{forest[i_case].name} != main{i_case+1}"
            leaves = forest[i_case].get_all_leaves()
            for i in range(len(leaves)):
                _action = {
                    'name': forest[i_case].name, 
                    # 'pred_model': args.model, 
                    'final_action': leaves[i].output,
                    'tree_index': leaves[i].tree_index,
                }
                actions.append(_action)
                
        name_to_result = evaluate_final_actions_batch(
            args, 
            model_name_or_path=args.eval_model, 
            data_path=args.input_path, 
            actions=actions,
            step=args.eval_step
        )

        with open(output_path_eval, 'w') as f:
            json.dump(name_to_result, f, indent=4)
    else:
        print(f"{output_path_eval} exists, skipping evaluate_final_actions and loading existing results.")
    
    calc_eval_metrics(output_path_eval, args.eval_step, verbose=False)

if __name__ == "__main__":
    main()
