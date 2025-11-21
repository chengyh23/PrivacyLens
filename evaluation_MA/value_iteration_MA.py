""" Johan C
"""
import os
import json
from typing import List, Dict
import numpy as np

from tree_utils import TreeNode, MultiAgentTree, Forest

def reward_model(eval_result):
    """
    Params:
        eval_result (dict):
            'leak_info': <bool>
            'secrete_jedgment': <>
    """
    return 1.0 if eval_result['leak_info'] else 0.0

def build_leaf_reward_map(flat_judgments):
    r"""
    Builds a mapping from (name, tree_position) to reward for all leaf nodes.
    Params:
        flat_judgments: 
            Dict[Dict]: {
                'main1': 
                    [ {
                        'tree_position': <pos>, 
                        'leak_info': bool, 
                        'secret_judgment': <sensitive_info_items>
                    }, {...} ]
                'mainX':
                    ...
            }
            List[Dict]: [
                {
                    'name': str,
                    'final_action': str,
                    'tree_index': <[0,0,0]>,
                    'eval_model': str,
                    'eval_result': {
                        'leak_info': bool,
                        'secret_judgment': <sensitive_info_items>
                    }
                }, 
                ...
            ]

    Returns:
        name_pos_to_R (dict): 
            A nested dictionary.
                First level: indexed by 'name' ('mainX').
                Second level: indexed by tree_position (as tuple) corresponding to a leaf trajectory in the tree.
                Value: 1 if 'leak_info' is True for that trajectory, else 0.
            Example:
                {
                    'main1': { 
                        (0, 0, 0): 0, 
                        (0, 0, 1): 1, 
                        ... 
                    },
                    'main2': { ... }
                }
        # TODO extend input List[Dict]
    """
    name_pos_to_R = {}
    if isinstance(flat_judgments, list):
        for judgment in flat_judgments:
            name = judgment['name']
            pos = tuple(judgment['tree_index'])
            if name not in name_pos_to_R:
                name_pos_to_R[name] = {}
            name_pos_to_R[name][pos] = reward_model(judgment['eval_result'])
    elif isinstance(flat_judgments, dict):
        for name, trajectories in flat_judgments.items():
            # The first level of the dict is keyed by each 'name'
            if name not in name_pos_to_R:
                name_pos_to_R[name] = {}
            for traj in trajectories:
                # The second level is keyed by the tree position tuple of each trajectory
                pos = tuple(traj['tree_position'])
                eval_result = {
                    'leak_info': traj['leak_info'], 
                    'secret_judgment': traj['secret_judgment']
                }
                name_pos_to_R[name][pos] = reward_model(eval_result)
    else:
        raise TypeError("Invalid type for flat_judgments in build_leaf_reward_map")
    return name_pos_to_R



# def set_V_on_leaves(tree: MultiAgentTree, pos_to_value):
#     """
#     Traverse all leaf nodes under root_node and set 'V' using pos_to_value mapping.
#     """
#     # Try to use the TreeNode method if available
#     get_leaves = tree.get_all_leaves()

#     for leaf in get_leaves:
#         pos = tuple(leaf.tree_index)
#         if pos not in pos_to_value:
#             raise KeyError(f"Leaf at tree_index={pos} not found in flat judgment mapping")
#         _V = pos_to_value[pos]
#         leaf.V = _V


def assign_reward_to_leaves(flat_judgments: List[Dict], forest: Forest):
    """
    in-place modification
    """
    
    for judgment in flat_judgments:
        name = judgment['name']
        pos = judgment['tree_index']
        tree = forest.get_tree_by_name(name)
        node = tree.get_node_by_index(pos)
        node.V = reward_model(judgment['eval_result']) # TODO need to encapsulate or not?
    # TODO check if all leaves are assigned a reward


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
    # print(f"Child vs: {child_vs} node: {node.type}")
    node.V = float(np.mean(child_vs))
    return node.V


def main(args):
    if os.path.exists(args.output_path):
        print("Output file already exists; skipping value iteration.")
        return
    # Reward assignment at leaves
    # forest = load_forest_from_json(args.action_path)
    forest = Forest.from_dict(args.action_path)
    with open(args.flat_judgment, "r") as f:
        flat_judgments = json.load(f)
    assign_reward_to_leaves(flat_judgments, forest)
    
    # Value Iteration
    for tree in forest.trees:
        propagate_expected_value(tree.root)
    
    forest.to_dict(args.output_path, verbose=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add reward values to nested tree structure")
    parser.add_argument("--action-path", required=True, help="Path to action (nested tree) JSON file")
    parser.add_argument("--flat-judgment", required=True, help="Path to flat judgment JSON file")
    parser.add_argument("--output-path", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    # forest = load_forest_from_json(args.action_path)

    # # Load flat-leaf judgments with leak_info
    # with open(args.flat_judgment, "r") as f:
    #     flat_judgments = json.load(f)
    # name_pos_to_R = build_leaf_reward_map(flat_judgments)
    
    # # Traverse and set 'V' for verifier leaves using the constructed mapping
    # for tree in forest:
    #     print(f"Processing tree: {tree.name}")
    #     # The following call will modify ex["tree"] IN PLACE
    #     set_V_on_leaves(tree, name_pos_to_R[tree.name])

    # save_forest_to_json(forest, args.output_path)
    # print(f"Set 'V' for verifier leaf nodes using flat judgment and saved to {args.output_path}")

    main(args)
