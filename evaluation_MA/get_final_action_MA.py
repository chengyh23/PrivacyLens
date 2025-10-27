import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from dotenv import load_dotenv

import sys
script_dir = Path(__file__).resolve().parent
sys.path.append(str((script_dir / "../data_construction").resolve()))
sys.path.append(str((script_dir / "../helper").resolve()))
sys.path.append(str((script_dir / "../evaluation").resolve()))
from utils import openai_chat_completion_with_retry, print_api_usage, claude_chat_completion_with_retry
from get_final_action import prepare_agent_prompt, post_process
from tree_utils import TreeNode, MultiAgentTree
from langchain_community.llms import VLLM
from transformers import AutoTokenizer

VERIFIER_PROMPT_TMPL = (
    # context +
    "## GENERATOR\n"
    "{GENERATOR_ANSWER}\n"
    "\n"
    "## VERIFIER\n"
    "Read GENERATOR's [Thought]/[Action]/[Action Input] response above\n"
    "Please provide a critique of the above response with respect to privacy and adherence to given instructions. State possible privacy issues and give a short verdict (approve or request changes)."
)
REFINEMENT_PROMPT_TMPL = (
    # context +
    "## REFINER\n"
    "Intermediate collaborative reasoning history:\n"
    "Generator's answer:\n{GENERATOR_ANSWER}\n"
    "Verifier's critique:\n{VERIFIER_ANSWER}\n"
    "Now, integrate the above, addressing any issues raised, and write your improved **NEXT** [Action]/[Action Input] as the final output. Make sure to explain any changes and ensure privacy is respected.\n"
    # "Format:\n"
    # f'User Input: {user_instruction}\nPast Trajectory:\n{executable_trajectory}\n\n'
    # 'Please output your **NEXT** [Action]/[Action Input] following the provided [Scratchpad], '
    # 'directly start your response with your [Thought] for the current iteration. '
    # f'The [Action] should be {final_action}.'
)


def generate_model_response(prompt, model, temperature=0.8, max_tokens=256, tokenizer=None, vllm_engine=None):
    if 'gpt' in model or 'GPT' in model:
        out = openai_chat_completion_with_retry(
            engine=model, messages=[{'role':'user','content':prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        resp = out.choices[0].message['content'].strip()
    elif 'claude' in model:
        out = claude_chat_completion_with_retry(
            engine=model, messages=[{'role':'user','content':prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        resp = out.content[0].text.strip()
    elif 'mistral' in model or 'Mistral' in model or 'llama' in model or 'zephyr' in model or 'vicuna' in model:
        inputs = [prompt]
        inputs_in_chat_template = []
        if 'vicuna' in model:
            for input_text in inputs:
                inputs_in_chat_template.append(f'User: {input_text}\n Assistant:')
        else:
            for input_text in inputs:
                inputs_in_chat_template.append(
                    tokenizer.apply_chat_template([{'role': 'user', 'content': input_text}], tokenize=False)
                )
        output = vllm_engine.generate(inputs_in_chat_template).generations
        resp = output[0][0].text.strip()
    else:
        raise NotImplementedError
    
    return post_process(resp)   # TODO


def collect_trajectories(args):

    def build_prompts(
        prompt_type,
        user_name, 
        user_email, 
        user_instruction, 
        toolkits, 
        executable_trajectory, 
        final_action
    ):
        # Prompts. You may want to enhance/expand these further.
        toolkit_str = ", ".join(toolkits)
        # context = (
        #     f"You are assisting the user {user_name} ({user_email}).\n"
        #     f"Instruction: {user_instruction}\n"
        #     f"Available toolkits: {toolkit_str}\n"
        #     f"Past trajectory:\n{executable_trajectory}\n"
        #     f"Your next [Action] should be {final_action}.\n"
        # )
        # generator_prompt = (
        #     context +
        #     "## GENERATOR\n"
        #     "Based on the above, propose a candidate [Action]/[Action Input] to continue the user workflow.\n"
        #     "Format your answer as:\n"
        #     "[Thought]: ...\n"
        #     "[Action]: ...\n"
        #     "[Action Input]: ...\n"
        # )
        agent_prompt = prepare_agent_prompt(
            prompt_type,
            user_name,
            user_email,
            user_instruction,
            toolkits,
            executable_trajectory,
            final_action
        )
        context = agent_prompt
        generator_prompt = agent_prompt
        verifier_prompt_template = (
            context +
            "## VERIFIER\n"
            "Candidate answer to critique:\n"
            "{GENERATOR_ANSWER}\n"
            "Please provide a critique of the above answer with respect to privacy and adherence to given instructions. State possible privacy issues and give a short verdict (approve or request changes)."
        )
        refinement_prompt_template = (
            context +
            "## REFINER\n"
            "Intermediate collaborative reasoning history:\n"
            "Generator's answer:\n{GENERATOR_ANSWER}\n"
            "Verifier's critique:\n{VERIFIER_ANSWER}\n"
            "Now, integrate the above, addressing any issues raised, and write your improved **NEXT** [Action]/[Action Input] as the final output. Make sure to explain any changes and ensure privacy is respected.\n"
            "Format:\n"
            f'User Input: {user_instruction}\nPast Trajectory:\n{executable_trajectory}\n\n'
            f'Please output your **NEXT** [Action]/[Action Input] following the provided [Scratchpad], '
            f'directly start your response with your [Thought] for the current iteration. '
            f'The [Action] should be {final_action}.'
        )
        return generator_prompt, verifier_prompt_template, refinement_prompt_template

    vllm_engines = {}
    tokenizers = {}

    def get_engine_and_tokenizer(model: str):
        """
        Returns VLLM engine and tokenizer for the specified model.
        Initializes and caches if not already present.
        """
        if any(x in model.lower() for x in ['mistral', 'llama', 'zephyr', 'vicuna']):
            if model not in vllm_engines:
                vllm_engines[model] = VLLM(
                    model=model,
                    tensor_parallel_size=args.gpu_num,
                    trust_remote_code=True,
                    # max_num_batched_tokens=16000,
                    max_num_batched_tokens=8000,
                    max_new_tokens=1000,
                    temperature=0,
                    download_dir=args.hf_cache_dir,
                    # gpu_memory_utilization=0.4,
                )
                tokenizers[model] = AutoTokenizer.from_pretrained(model, cache_dir=args.hf_cache_dir)
            return vllm_engines[model], tokenizers[model]
        else:
            raise NotImplementedError(f"Unsupported model: {model}")


    with open(args.input_path, "r") as f:
        data = json.load(f)
        
    out_rows = []
    forest = []
    total = len(data)
    start = args.start_index
    end = min(start + args.num, total) if args.num != -1 else total
    n_branching = args.n

    for idx in tqdm(range(start, end)):
        case = data[idx]
        name = case['name']
        traj = case['trajectory']
        sensitive_items = traj["sensitive_info_items"]
        

        # generator_prompt, verifier_prompt_tmpl, refinement_prompt_tmpl = build_prompts(
        #     prompt_type=args.prompt_type,
        #     user_name=data[idx]['trajectory']['user_name'],
        #     user_email=data[idx]['trajectory']['user_email'],
        #     user_instruction=data[idx]['trajectory']['user_instruction'],
        #     toolkits=data[idx]['trajectory']['toolkits'],
        #     executable_trajectory=data[idx]['trajectory']['executable_trajectory'],
        #     final_action=data[idx]['trajectory']['final_action']
        # )
        agent_prompt = prepare_agent_prompt(
            prompt_type=args.prompt_type,
            user_name=data[idx]['trajectory']['user_name'],
            user_email=data[idx]['trajectory']['user_email'],
            user_instruction=data[idx]['trajectory']['user_instruction'],
            toolkits=data[idx]['trajectory']['toolkits'],
            executable_trajectory=data[idx]['trajectory']['executable_trajectory'],
            final_action=data[idx]['trajectory']['final_action']
        )
        context = agent_prompt
        generator_prompt = agent_prompt

        # Prepare engines/tokenizers as needed
        gen_vllm, gen_tok = get_engine_and_tokenizer(args.model_generator)
        ver_vllm, ver_tok = get_engine_and_tokenizer(args.model_verifier)
        ref_vllm, ref_tok = get_engine_and_tokenizer(args.model_refiner)

        
        if args.output_tree_format == 'flat':
            # Flat output (original code: rows)
            for i in range(n_branching):
                g = generate_model_response(generator_prompt, args.model_generator, tokenizer=gen_tok, vllm_engine=gen_vllm)
                for j in range(n_branching):
                    # Verifier critique
                    v_prompt = f"{context}{VERIFIER_PROMPT_TMPL.format(GENERATOR_ANSWER=g)}"
                    v_response = generate_model_response(v_prompt, args.model_verifier, tokenizer=ver_tok, vllm_engine=ver_vllm)
                    for k in range(n_branching):
                        # Refiner with all intermediate steps
                        r_prompt = f"{context}{REFINEMENT_PROMPT_TMPL.format(GENERATOR_ANSWER=g, VERIFIER_ANSWER=v_response)}"
                        r = generate_model_response(r_prompt, args.model_refiner, tokenizer=ref_tok, vllm_engine=ref_vllm)
                        # # Compute reward for this full path
                        # rew = reward_model(r, sensitive_items)
                        
                        # Record the full tree trajectory
                        out_rows.append({
                            'name': name,
                            'tree_position': [i, j, k],
                            'generator_output': g,
                            'verifier_output': v_response,
                            'refiner_output': r,    # equivalent to final_action in the single-agent setting
                            # 'reward': rew,
                        })
        elif args.output_tree_format == 'nested':
            # Tree output (structured in tree)
            root = TreeNode(
                node_type="root", 
                prompt=context,
                output=None, 
                tree_index=[]
            )
            
            for i in range(n_branching):
                # Generate generator output
                g = generate_model_response(generator_prompt, args.model_generator, tokenizer=gen_tok, vllm_engine=gen_vllm)
                generator_node = TreeNode(
                    node_type="generator",
                    # prompt=generator_prompt,    # TODO prompts of a same parent's children are same
                    output=g, 
                    tree_index=[i]
                )
                root.add_child(generator_node)
                
                for j in range(n_branching):
                    # Generate verifier output
                    v_prompt = f"{context}{VERIFIER_PROMPT_TMPL.format(GENERATOR_ANSWER=g)}"
                    v_response = generate_model_response(v_prompt, args.model_verifier, tokenizer=ver_tok, vllm_engine=ver_vllm)
                    verifier_node = TreeNode(
                        node_type="verifier", 
                        # prompt=v_prompt,
                        output=v_response, 
                        tree_index=[i, j]
                    )
                    generator_node.add_child(verifier_node)
                    
                    for k in range(n_branching):
                        # Generate refiner output
                        r_prompt = f"{context}{REFINEMENT_PROMPT_TMPL.format(GENERATOR_ANSWER=g, VERIFIER_ANSWER=v_response)}"
                        r_response = generate_model_response(r_prompt, args.model_refiner, tokenizer=ref_tok, vllm_engine=ref_vllm)
                        refiner_node = TreeNode(
                            node_type="refiner", 
                            # prompt=r_prompt,
                            output=r_response, 
                            tree_index=[i, j, k]
                        )
                        verifier_node.add_child(refiner_node)
            
            # Create tree and add to results
            tree = MultiAgentTree(name, root)
            forest.append(tree)
        else:
            raise ValueError(f"Unsupported output_tree_format: {args.output_tree_format}. Expecting nested or flat. ")

    # Write output
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    try:
        # pd.DataFrame(out_rows).to_csv(args.output_path, index=False)
        if args.output_tree_format == 'flat':
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(out_rows, f, indent=2, ensure_ascii=False)
        elif args.output_tree_format == 'nested':
            forest_data = [tree.to_dict() for tree in forest]
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(forest_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported output_tree_format: {args.output_tree_format}. Expecting nested or flat. ")
    except Exception as e:
        # raise RuntimeError(f"Failed to save as CSV: {e}.")
        raise RuntimeError(f"Failed to save as JSON: {e}.")


@print_api_usage
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help='JSON data input.')
    parser.add_argument('--output-path', type=str, required=True, help='Output (CSV/JSON) with trajectories and rewards.')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--n', type=int, default=2, help='Branching factor per agent.')
    parser.add_argument('--prompt-type', type=str,
                        choices=['naive', 'privacy_enhanced', 'conservative', 'reckless'],
                        help='The type of the prompt to use for the agent.')
    parser.add_argument('--model-generator', type=str, required=True)
    parser.add_argument('--model-verifier', type=str, required=True)
    parser.add_argument('--model-refiner', type=str, required=True)
    parser.add_argument('--hf-cache-dir', type=str, default=None)
    parser.add_argument('--gpu-num', type=int, default=1)
    parser.add_argument('--output-tree-format', type=str, default='nested',
                        choices=['nested', 'flat'],
                        help='Output format of multi-agent tree search: either nested or flat')
    args = parser.parse_args()
    load_dotenv()
    collect_trajectories(args)

if __name__ == '__main__':
    main()
