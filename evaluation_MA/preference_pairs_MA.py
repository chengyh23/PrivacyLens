import json
import os
import numpy as np
import argparse
from torch.utils.data import dataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from tree_utils import TreeNode, MultiAgentTree, build_tree_from_flat_data, Forest
from get_final_action_MA import VERIFIER_PROMPT_TMPL, REFINEMENT_PROMPT_TMPL

from data_pipeline_aug.gen_preference_pairs import cartesian_product

def collect_preference_pairs_from_tree(tree: MultiAgentTree, threshold=0.5, verbose=True):
    """
    Collect preference pairs for both verifier and refiner roles.

    Returns:
        - verifier_preference_pairs: list of dicts.
        - refiner_preference_pairs: list of dicts.
    """
    
    preference_pairs_V = []
    preference_pairs_R = []

    name, root = tree.name, tree.root
    context = root.prompt
    # VERIFIER PAIRS: for each generator node, consider verifier children
    for nodeG in root.children:
    # for i, nodeG in enumerate(root.children):
        verifiers = [(j, nodeV) for j, nodeV in enumerate(nodeG.children)]
        v_prompt = (
            f"{context}"
            f"{VERIFIER_PROMPT_TMPL.format(GENERATOR_ANSWER=nodeG.output)}" 
        )

        # # Use mean of verifier node_j.V across generator node's children
        # if threshold is None:
        #     verifier_vs = [node_j.V for _, node_j in verifier_children]
        #     verifier_thresh = np.mean(verifier_vs)
        # else:
        #     verifier_thresh = threshold
        verifier_thresh = threshold

        positive_verifiers_response = [node_j.output for _, node_j in verifiers if node_j.V >= verifier_thresh]
        negative_verifiers_response = [node_j.output for _, node_j in verifiers if node_j.V < verifier_thresh]
        case_pref_pairs_verifier = cartesian_product(positive_verifiers_response, negative_verifiers_response, name, v_prompt)
        preference_pairs_V += case_pref_pairs_verifier
        if verbose:
            print(f'{name}:{nodeG.tree_index}: {len(positive_verifiers_response):<2} x {len(negative_verifiers_response):<2} = {len(case_pref_pairs_verifier):<4}')

        
        # REFINER PAIRS: for each verifier node, consider refiner children
        for nodeV in nodeG.children:
            refiners = [(k, child) for k, child in enumerate(nodeV.children)]
            r_prompt = (
                f"{context}"
                f"{REFINEMENT_PROMPT_TMPL.format(GENERATOR_ANSWER=nodeG.output, VERIFIER_ANSWER=nodeV.output)}"
            )

            # if threshold is None:
            #     thresh = np.mean([child.V for _, child in child_vs])
            # else:
            #     thresh = threshold
            refiner_thresh = threshold

            positive_refiner_response = [child.output for _, child in refiners if child.V >= refiner_thresh]
            negative_refiner_response = [child.output for _, child in refiners if child.V < refiner_thresh]
            case_pref_pairs_refiner = cartesian_product(positive_refiner_response, negative_refiner_response, name, r_prompt)
            preference_pairs_R += case_pref_pairs_refiner
            if verbose:
                print(f'{name}:{nodeV.tree_index}: {len(positive_refiner_response):<2} x {len(negative_refiner_response):<2} = {len(case_pref_pairs_refiner):<4}')

            
    return preference_pairs_V, preference_pairs_R

def parse_args():
    parser = argparse.ArgumentParser(description='Generate preference pairs for verifier and refiner training')
    parser.add_argument('--input_path', 
                       required=True,
                       help='Path to the input nested tree JSON file')
    parser.add_argument('--output_path_V', 
                       default='data_pipeline_MA/pref_pairs_verifier.json',
                       help='Path to save verifier preference pairs')
    parser.add_argument('--output_path_R', 
                       default='data_pipeline_MA/pref_pairs_refiner.json',
                       help='Path to save refiner preference pairs')
    parser.add_argument('--threshold', 
                       type=float, 
                       default=0.5,
                       help='Threshold for determining positive vs negative examples')
    
    return parser.parse_args()

def main(verbose: bool = False):
    args = parse_args()
    input_path = args.input_path
    output_path_V = args.output_path_V
    output_path_R = args.output_path_R
    threshold = args.threshold
    
    # if os.path.exists(output_path_V) and os.path.exists(output_path_R):
    #     # INSERT_YOUR_CODE
    #     with open(output_path_V, 'r') as f_v:
    #         datasetV = json.load(f_v)
    #     with open(output_path_R, 'r') as f_r:
    #         datasetR = json.load(f_r)
    #     print("Skip: output files already exist.")
    #     print(f"Generated #pairs: verifier {len(datasetV)} pairs; refiner {len(datasetR)} pairs")
    #     return

    # if not os.path.exists(input_path):
    #     raise FileNotFoundError(
    #         f"The nested tree file '{input_path}' does not exist. Please generate it first."
    #     )
    
    print(f"Loading forest from: {input_path}")
    forest = Forest.from_dict(input_path)

    datasetV, datasetR = [], []
    test_case_names = []
    for tree in forest.trees:
        verifier_pairs, refiner_pairs = collect_preference_pairs_from_tree(tree, threshold, verbose=False)
        datasetV += verifier_pairs
        datasetR += refiner_pairs
        if len(verifier_pairs)==0 and len(refiner_pairs)==0:
            test_case_names.append(tree.name)
        if verbose:
            print(f"{tree.name}: {len(verifier_pairs)} verifier & {len(refiner_pairs)} refiner pairs added.")
    
    # Save preference pairs
    if os.path.exists(output_path_V):
        with open(output_path_V, "r", encoding="utf-8") as fin:
            pref_pairs_existing = json.load(fin)
        print(f"Already exists: {output_path_V}\n(Total # preference pairs: {len(pref_pairs_existing)})")
    else:
        print(f"Saving verifier pairs to: {output_path_V}")
        with open(output_path_V, 'w') as f:
            json.dump(datasetV, f, indent=2)
        print(f"Generated #pairs: verifier {len(datasetV)} pairs; refiner {len(datasetR)} pairs")
    
    if os.path.exists(output_path_R):
        with open(output_path_R, "r", encoding="utf-8") as fin:
            pref_pairs_existing = json.load(fin)
        print(f"Already exists: {output_path_R}\n(Total # preference pairs: {len(pref_pairs_existing)})")
    else:
        print(f"Saving refiner pairs to: {output_path_R}")
        with open(output_path_R, 'w') as f:
            json.dump(datasetR, f, indent=2)
    
    # Save test cases
    test_case_path = os.path.commonprefix([output_path_V, output_path_R]) + "empty_cases.txt"
    if os.path.exists(test_case_path):
        print(f"Already exists: {test_case_path}\n(Total # test cases: {len(test_case_names)})")
    else:
        with open(test_case_path, "w", encoding="utf-8") as fout:
            for name in test_case_names:
                fout.write(f"{name}\n")
        print(f"Wrote out test cases' names to {test_case_path}")
    

if __name__ == '__main__':
    main()
