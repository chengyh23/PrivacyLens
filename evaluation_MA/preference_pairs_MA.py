import json
import os
import numpy as np
import argparse
from torch.utils.data import dataset

# import sys
# from pathlib import Path
# script_dir = Path(__file__).resolve().parent
# sys.path.append(str((script_dir / "../").resolve()))

from tree_utils import TreeNode, MultiAgentTree, build_tree_from_flat_data, load_forest_from_json
from get_final_action_MA import VERIFIER_PROMPT_TMPL, REFINEMENT_PROMPT_TMPL


def collect_preference_pairs_from_tree(tree: MultiAgentTree, threshold=0.5):
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
    for nodeG in root.children:
    # for i, nodeG in enumerate(root.children):
        # VERIFIER PAIRS: for each generator node, consider verifier children
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

        positive_verifiers = [node_j for _, node_j in verifiers if node_j.V >= verifier_thresh]
        negative_verifiers = [node_j for _, node_j in verifiers if node_j.V < verifier_thresh]
        # pairwise Cartesian product across verifier nodes
        for pos in positive_verifiers:
            for neg in negative_verifiers:
                prompt = v_prompt
                chosen = pos["output"]
                rejected = neg["output"]
                if not chosen or not rejected:
                    raise ValueError(f"Missing chosen or rejected value in verifier/refiner output: chosen={chosen!r}, rejected={rejected!r}")
                output = {
                    "name": name,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
                preference_pairs_V.append(output)

        for nodeV in nodeG.children:
            # REFINER PAIRS: for each verifier node, consider refiner children
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

            positive_refiner = [child for _, child in refiners if child.V >= refiner_thresh]
            negative_refiner = [child for _, child in refiners if child.V < refiner_thresh]
            for pos in positive_refiner:
                for neg in negative_refiner:
                    prompt = r_prompt
                    chosen = pos["output"]
                    rejected = neg["output"]
                    if not chosen or not rejected:
                        raise ValueError(f"Missing chosen or rejected value in verifier/refiner output: chosen={chosen!r}, rejected={rejected!r}")
                    output = {
                        "name": name,
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                    preference_pairs_R.append(output)
    return preference_pairs_V, preference_pairs_R

def main():
    parser = argparse.ArgumentParser(description='Generate preference pairs for verifier and refiner training')
    parser.add_argument('--input_path', 
                       default='data_pipeline_MA/predictions/mistral-7b-instruct-v02-NESTED-with-V.json',
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
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path_V = args.output_path_V
    output_path_R = args.output_path_R
    threshold = args.threshold
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"The nested tree file '{input_path}' does not exist. Please generate it first."
        )
    
    print(f"Loading forest from: {input_path}")
    forest = load_forest_from_json(input_path)

    datasetV, datasetR = [], []
    for tree in forest:
        verifier_pairs, refiner_pairs = collect_preference_pairs_from_tree(tree, threshold)
        datasetV += verifier_pairs
        datasetR += refiner_pairs
    
    # Save the pairs
    print(f"Saving verifier pairs to: {output_path_V}")
    with open(output_path_V, 'w') as f:
        json.dump(datasetV, f, indent=2)
    
    print(f"Saving refiner pairs to: {output_path_R}")
    with open(output_path_R, 'w') as f:
        json.dump(datasetR, f, indent=2)
    
    print(f"Generated {len(datasetV)} verifier pairs and {len(datasetR)} refiner pairs")

if __name__ == '__main__':
    main()
