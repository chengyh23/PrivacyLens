""" Johan C
"""

import json
import numpy as np
from tree_utils import load_forest_from_json, TreeNode
from tree_utils import build_tree_from_flat_data


def propagate_expected_value(node: TreeNode):
    """
    Recursively computes the expected value V for all nodes in the tree.
    Leaf nodes take their reward as V; non-leaf nodes take mean of children's V.
    """
    if not node.children:
        # Leaf node: already has V set (should be reward from leaf trajectory)
        return node.V
    child_vs = [propagate_expected_value(child) for child in node.children]
    if not child_vs:
        raise ValueError(f"Tree node '{node.type}' has no children; cannot compute mean.")
    print(f"Child vs: {child_vs} node: {node.type}")
    node.V = float(np.mean(child_vs))
    return node.V

def annotate_tree_dict_with_values(node: TreeNode, node_dict: dict):
    """
    Recursively updates the provided node_dict by writing the latest V value at each node.
    """
    node_dict['V'] = node.V
    if 'children' in node_dict and node.children:
        for idx, child in enumerate(node.children):
            # Children in node_dict are a list, so map by index
            annotate_tree_dict_with_values(child, node_dict['children'][idx])


def value_iteration(input_path: str, output_path: str):
    """
    Performs value iteration on a forest of nested trees (read from JSON), propagates
    values up the tree, and writes V-annotated trees back to a file.
    
    Args:
        input_path: Path to input JSON file with nested trees
        output_path: Path to output JSON file with computed values
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"The nested tree file '{input_path}' does not exist. Please generate it first."
        )
    
    # Load the forest of tree dicts from file
    with open(input_path, 'r', encoding='utf-8') as f:
        forest_dicts = json.load(f)

    trees = load_forest_from_json(input_path)

    # Propagate value (V) in each tree
    for tree in trees:
        propagate_expected_value(tree.root)

    # Write back with updated 'V' fields everywhere
    for tree, orig_dict in zip(trees, forest_dicts):
        annotate_tree_dict_with_values(tree.root, orig_dict["tree"])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(forest_dicts, f, indent=2, ensure_ascii=False)
    
    print(f"Value iteration completed. Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Perform value iteration on multi-agent trees")
    parser.add_argument("--input", required=True, help="Input JSON file with nested trees")
    parser.add_argument("--output", required=True, help="Output JSON file with computed values")
    
    args = parser.parse_args()
    
    # try:
    value_iteration(args.input, args.output)
    # except Exception as e:
    #     print(f"Error during value iteration: {e}")
    #     exit(1)