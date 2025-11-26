import argparse
import json
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from evaluation.evaluate_final_action import evaluate_final_actions_batch
from evaluation.get_final_action import load_original_data, get_final_actions_batch, prepare_agent_prompt_to_leak, prepare_agent_prompt_with_example, generate_model_response, prepare_agent_prompt_with_full_hints

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
        generations = get_final_actions_batch(
            args, pred_prompt_func, pred_model_name_or_path,
            specific_case_name=cases_to_process,
            n_sample_per_case=n_sample_per_case,
        )

        names, final_actions = generations['name'], generations['final_action']
        actions = [{"name": name, "final_action": final_action} for name, final_action in zip(names, final_actions)]

        name_and_result = evaluate_final_actions_batch(
            args, eval_model_name_or_path, args.input_path, actions, args.eval_step,
        )
        
        # Save evaluation results
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        _fname, _extension = os.path.splitext(args.output_path)
        output_path = f"{_fname}-round{_rounds}{'.' + mode if mode in {'hard', 'easy'} else ''}{_extension}"

        if args.eval_step == 'extract_secret':
            for i in range(len(data)):
                data[i]['trajectory']['sensitive_info_items'] = name_and_result[data[i]['name']]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            with open(output_path, 'w') as f:
                json.dump(name_and_result, f, indent=4)

        # Group evaluations results by name, and privacy leaked/preserved
        name_indices = defaultdict(list)
        for idx, entry in enumerate(name_and_result):
            name_indices[entry["name"]].append(idx)

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
                        help='Path of the PrivacyLens data in json format.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the probing results in csv format.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the cases to evaluate.')
    parser.add_argument('--num-case', type=int, default=1,
                        help='Number of cases to evaluate. If -1, evaluate all remaining cases.')
    parser.add_argument('--specific-case-name', type=str, nargs='*', default=None,
                        help='If not None, only evaluate the case with the given name.')
    parser.add_argument('--prompt-type', type=str,
                        choices=['naive', 'privacy_enhanced', 'conservative', 'reckless'],
                        help='The type of the prompt to use for the agent.')
    parser.add_argument('--pred-model', type=str, required=True,
                        help='The model to use for generating action.')
    parser.add_argument('--eval-model', type=str, required=True,
                        help='The model to use for evaluating action.')
    parser.add_argument('--eval-step', type=str,
                        choices=['extract_secret', 'judge_leakage', 'helpfulness'])
    parser.add_argument('--n-sample-per-case', type=int, required=True,
                        help='N generations per case.')
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs to use for VLLM.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.')
    parser.add_argument('--hf-cache-dir', type=str,
                        help='The cache directory for the Hugging Face model.')

    return parser.parse_args()

if __name__ == '__main__':
    args = prepare_args()

    # hard_cases, easy_cases = iterative_get_and_eval_final_actions(
    #     args, 
    #     pred_model_name_or_path=args.pred_model, 
    #     eval_model_name_or_path=args.eval_model, 
    #     specific_case_name=args.specific_case_name, 
    #     num_case=args.num_case, 
    #     n_sample_per_case=args.n_sample_per_case,
    #     mode="normal"
    # )
    # hard_cases2 = iterative_get_and_eval_final_actions(
    #     args, 
    #     pred_model_name_or_path=args.pred_model, 
    #     eval_model_name_or_path=args.eval_model, 
    #     specific_case_name=hard_cases, 
    #     num_case=args.num_case, 
    #     n_sample_per_case=args.n_sample_per_case,
    #     mode="hard"
    # )
    easy_cases = ['main408', 'main428']
    easy_cases2 = iterative_get_and_eval_final_actions(
        args, 
        pred_model_name_or_path=args.pred_model, 
        eval_model_name_or_path=args.eval_model, 
        specific_case_name=easy_cases, 
        num_case=args.num_case, 
        n_sample_per_case=args.n_sample_per_case,
        mode="easy",
        starting_round=2
    )


    