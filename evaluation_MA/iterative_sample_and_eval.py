import argparse
import os
import json
from collections import defaultdict
import random
import numpy as np
from dotenv import load_dotenv

import torch
from tqdm import tqdm
from get_final_action_MA import collect_trajectories_batch
from evaluate_final_action_MA import eval_final_actions
from value_iteration_MA import assign_reward_to_leaves, propagate_expected_value
from evaluation.evaluate_final_action import evaluate_final_actions_batch, prepare_prompt_to_identify_sensitive_information, prepare_prompt_to_identify_leakage, prepare_prompt_to_evaluate_helpfulness
from evaluation.get_final_action import load_original_data, prepare_agent_prompt_to_leak, prepare_agent_prompt_with_example, generate_model_response, prepare_agent_prompt_with_full_hints
 

def iterative_get_and_eval_final_actions(
    args, 
    pred_model_name_or_path: str, 
    eval_model_name_or_path: str, 
    specific_case_name: str = None,
    num_case: int = None, 
    n_sample_per_case: int = 1,
    mode: str = "normal",  # "normal", "hard", "easy"
    starting_round: int = 0,
):
    """
    Evaluates the final actions for PrivacyLens cases depending on the mode.

    Args:
        args: argparse.Namespace
        pred_model_name_or_path (str)
        eval_model_name_or_path (str)
        specific_case_name (Optional[list[str]], optional): Only evaluate these case names (overrides num_case/start_index).
        num_case (int, optional): Number of cases to evaluate.
        n_sample_per_case (int, optional): Number of generations per case.
        mode (str): "normal", "hard", or "easy"
            - "normal": returns both hard and easy cases
            - "hard": focuses on hard cases (those only with privacy leaks)
            - "easy": focuses on easy cases (those only with privacy preserved)

    Returns:
        Depending on mode:
            "normal": (hard_case_names: List[str], easy_case_names: List[str])
            "hard": hard_case_names: List[str]
            "easy": easy_case_names: List[str]
    """
    # Select the prompt function based on mode
    if mode == "normal":
        pred_prompt_func = prepare_agent_prompt_with_example
    elif mode == "hard":
        pred_prompt_func = prepare_agent_prompt_with_full_hints
    elif mode == "easy":
        pred_prompt_func = prepare_agent_prompt_to_leak
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Load data
    data = load_original_data(args.input_path)
    # Determine cases to process
    if specific_case_name:
        cases_to_process = specific_case_name.copy()
    else:
        assert num_case is not None
        if num_case == -1:
            end_index = len(data)
        else:
            end_index = min(args.start_index + num_case, len(data))
        cases_to_process = [data[i]['name'] for i in tqdm(range(args.start_index, end_index))]

    _rounds = starting_round

    # Loop until all are processed or max rounds
    while True:
        forest = collect_trajectories_batch(
            args, pred_prompt_func, pred_model_name_or_path,
            specific_case_name=cases_to_process,
            n_sample_per_case=n_sample_per_case,
        )

        actions = []
        for i_case in range(len(forest)):
            assert forest[i_case].name == f'main{i_case+1}', f"{forest[i_case].name} != main{i_case+1}"
            leaves = forest[i_case].get_all_leaves()
            for i in range(len(leaves)):
                _action = {
                    'name': forest[i_case].name,
                    'final_action': leaves[i],
                    'tree_index': leaves[i].tree_index,
                }
                actions.append(_action)

        # Eval actions
        flat_judgments = evaluate_final_actions_batch(
            args, eval_model_name_or_path, args.input_path, actions, args.eval_step,
        )
        
        # Value iteration
        assign_reward_to_leaves(flat_judgments, forest)
        for tree in forest.trees:
            propagate_expected_value(tree.root)

        # Save evaluation results
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        _fname, _extension = os.path.splitext(args.output_path)
        output_path = f"{_fname}-round{_rounds}{'.' + mode if mode in {'hard', 'easy'} else ''}{_extension}"

        forest.to_dict(output_path, verbose=True)
        

        # Group evaluations results by name, and privacy leaked/preserved
        name_indices = defaultdict(list)
        for idx, entry in enumerate(name_and_result):
            name_indices[entry["name"]].append(idx)

            
        for tree in forest.trees:
            verifier_pairs, refiner_pairs = collect_preference_pairs_from_tree(tree, threshold, verbose=True)


        _n_case_prev = len(cases_to_process)
        cases_to_process_next = []
        hard_case_name, easy_case_name = [], []
        for name, indices in name_indices.items():
            pos_actions, neg_actions = [], []
            for index in indices:
                entry = name_and_result[index]
                assert name == entry["name"]
                if entry["eval_result"]["leak_info"] is True:
                    neg_actions.append(entry["final_action"])
                elif entry["eval_result"]["leak_info"] is False:
                    pos_actions.append(entry["final_action"])
                else:
                    raise ValueError(f"Unexpected value for leak_info: {entry['eval_result']['leak_info']}")
            print(f"{name}: {len(neg_actions)} x {len(pos_actions)}")
            
            # Logic branches based on mode
            if mode == "normal":
                if len(neg_actions) == 0:
                    cases_to_process_next.append(name)
                    easy_case_name.append(name)
                elif len(pos_actions) == 0:
                    cases_to_process_next.append(name)
                    hard_case_name.append(name)
                # Both positive and negative: not added to cases_to_process_next
            elif mode == "hard":
                # Only process cases where all generations leak (no positive found)
                if len(pos_actions) == 0:
                    cases_to_process_next.append(name)
                    hard_case_name.append(name)
            elif mode == "easy":
                # Only process cases where all generations are safe (no negative found)
                if len(neg_actions) == 0:
                    cases_to_process_next.append(name)
                    easy_case_name.append(name)

        _n_case_current = len(cases_to_process_next)
        print(f"[{mode}.round{_rounds}] # Cases to be processed: {_n_case_prev} -> {_n_case_current}")

        _rounds += 1
        if _n_case_current == 0:
            if mode == "normal":
                print("All cases now have been collected some preference pairs")
            elif mode == "hard":
                print("All hard cases now have been collected some positive responses")
            elif mode == "easy":
                print("All easy cases now have been collected some negative responses")
            break
        MAX_OROUND = 2
        if _rounds > MAX_OROUND - 1:
            if mode == "normal":
                print(f"After {MAX_OROUND} rounds, still fail to collect preference pairs from:")
                print("- Hard case:", hard_case_name)
                print("- Easy case:", easy_case_name)
            elif mode == "hard":
                print(f"After {MAX_OROUND} rounds, still fail to collect positive response from:", cases_to_process_next)
            elif mode == "easy":
                print(f"After {MAX_OROUND} rounds, still fail to collect negative response from:", cases_to_process_next)
            break
        cases_to_process = cases_to_process_next

    if mode == "normal":
        return hard_case_name, easy_case_name
    elif mode == "hard":
        return hard_case_name
    elif mode == "easy":
        return easy_case_name

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to the input data in JSON format.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the sampled and evaluation results.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the cases to process.')
    parser.add_argument('--num-case', type=int, default=1,
                        help='Number of cases to process. If -1, process all remaining cases.')
    parser.add_argument('--specific-case-name', type=str, default=None,
                        help='If specified, only process the case with this name.')
    parser.add_argument('--prompt-type', type=str, default='naive',
                        choices=['naive', 'privacy_enhanced', 'conservative', 'reckless'],
                        help='Prompt type to use for sampling actions.')
    parser.add_argument('--pred-model', type=str, required=True,
                        help='Model to use for generating action samples.')
    parser.add_argument('--eval-model', type=str, required=True,
                        help='Model to use for evaluation/judgment.')
    parser.add_argument('--eval-step', type=str, required=True,
                        choices=['judge_leakage', 'helpfulness'],
                        help='Step of evaluation.')
    parser.add_argument('--n-sample-per-case', type=int, default=1,
                        help='Number of action samples per case.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate to model(s).')
    parser.add_argument('--hf-cache-dir', type=str, default=None,
                        help='HuggingFace cache directory (optional).')
    return parser.parse_args()

def load_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    args = prepare_args()
    load_dotenv()
    seed_everything(0)

    # 1. Step 1: Sample actions for each case (if not existing)
    if not os.path.exists(args.output_path):
        # Import here to avoid top-level dependencies for unused pipeline

        actions = get_final_actions(
            args, 
            model_name_or_path=args.pred_model, 
            num_case=args.num_case, 
            n_sample_per_case=args.n_sample_per_case
        )
        # Format for storage
        sampled = []
        for name, action in zip(actions['name'], actions['final_action']):
            sampled.append({
                'id': name,
                'pred_model': args.pred_model,
                'final_action': action
            })
        save_json(sampled, args.output_path)
        print(f"Sampled {len(sampled)} cases written to {args.output_path}")
    else:
        print(f"{args.output_path} exists, skipping generation and loading existing results.")

    # 2. Step 2: Judge/evaluate the actions (append results to output)

    # Loads sampled actions
    sampled = load_data(args.output_path)
    # Since judge output is same file, we overwrite with judged results
    judged = eval_final_actions(
        args,
        cases=sampled,
        eval_model=args.eval_model,
        eval_step=args.eval_step,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    save_json(judged, args.output_path)
    print(f"Evaluated {len(judged)} cases and wrote results to {args.output_path}")

if __name__ == '__main__':
    main()
