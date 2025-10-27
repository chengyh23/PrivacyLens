import json

from tree_utils import load_forest_from_json, save_forest_to_json
from tree_utils import MultiAgentTree


def build_leaf_reward_map(flat_judgments):
    """
    Builds a mapping from (name, tree_position) to reward for all leaf nodes.

    Returns:
        name_pos_to_R (dict): 
            A nested dictionary.
            - First level: indexed by 'name' (typically the instance, conversation, or sample name).
            - Second level: indexed by tree_position (as tuple) corresponding to a leaf trajectory in the tree.
            - Value: 1 if 'leak_info' is True for that trajectory, else 0.
            Example:
                {
                    'main1': { (0, 0, 0): 0, (0, 0, 1): 1, ... },
                    'main2': { ... }
                }
    """
    name_pos_to_R = {}
    for name, trajectories in flat_judgments.items():
        # The first level of the dict is keyed by each 'name'
        if name not in name_pos_to_R:
            name_pos_to_R[name] = {}
        for traj in trajectories:
            # The second level is keyed by the tree position tuple of each trajectory
            pos = tuple(traj['tree_position'])
            leak_info = traj['leak_info']
            # Each value is 1 if leak_info is True at that leaf, else 0
            name_pos_to_R[name][pos] = 1 if leak_info else 0
    return name_pos_to_R



def set_V_on_leaves(tree: MultiAgentTree, pos_to_value):
    """
    Traverse all leaf nodes under root_node and set 'V' using pos_to_value mapping.
    """
    # Try to use the TreeNode method if available
    get_leaves = tree.get_all_leaves()

    for leaf in get_leaves:
        pos = tuple(leaf.tree_index)
        if pos not in pos_to_value:
            raise KeyError(f"Leaf at tree_index={pos} not found in flat judgment mapping")
        _V = pos_to_value[pos]
        leaf.V = _V
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add reward values to nested tree structure")
    parser.add_argument("--action-path", required=True, help="Path to action (nested tree) JSON file")
    parser.add_argument("--flat-judgment", required=True, help="Path to flat judgment JSON file")
    parser.add_argument("--output-path", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    forest = load_forest_from_json(args.action_path)

    # Load flat-leaf judgments with leak_info
    with open(args.flat_judgment, "r") as f:
        flat_judgments = json.load(f)
    name_pos_to_R = build_leaf_reward_map(flat_judgments)
    
    # Traverse and set 'V' for verifier leaves using the constructed mapping
    for tree in forest:
        print(f"Processing tree: {tree.name}")
        # The following call will modify ex["tree"] IN PLACE
        set_V_on_leaves(tree, name_pos_to_R[tree.name])

    save_forest_to_json(forest, args.output_path)
    print(f"Set 'V' for verifier leaf nodes using flat judgment and saved to {args.output_path}")
