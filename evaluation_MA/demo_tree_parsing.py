#!/usr/bin/env python3
"""
Demo script showing how to parse and work with the nested tree structure.
"""

import json
from tree_utils import MultiAgentTree, TreeNode

def demo_tree_parsing():
    """Demonstrate tree parsing and analysis."""
    
    # Load example tree
    with open('example_tree_structure.json', 'r') as f:
        tree_data = json.load(f)
    
    tree = MultiAgentTree.from_dict(tree_data)
    
    print("=== Tree Structure Analysis ===")
    print(f"Case: {tree.name}")
    
    # Get all leaves
    leaves = tree.get_all_leaves()
    print(f"\nTotal leaf nodes: {len(leaves)}")
    
    # Show all paths from root to leaves
    print("\n=== All Paths ===")
    for i, leaf in enumerate(leaves):
        path = leaf.get_path_to_root()
        print(f"\nPath {i+1}: {' -> '.join(f'{node.type}[{node.tree_index}]' for node in path)}")
        print(f"  Generator: {path[1].output[:50]}..." if len(path) > 1 else "  No generator")
        print(f"  Verifier: {path[2].output[:50]}..." if len(path) > 2 else "  No verifier") 
        print(f"  Refiner: {path[3].output[:50]}..." if len(path) > 3 else "  No refiner")
    
    # Get leaf outputs in flat format (equivalent to current implementation)
    print("\n=== Leaf Outputs (Flat Format) ===")
    leaf_outputs = tree.get_leaf_outputs()
    for output in leaf_outputs:
        print(f"\nTree Index: {output['tree_index']}")
        print(f"Generator: {output['generator_output'][:50]}...")
        print(f"Verifier: {output['verifier_output'][:50]}...")
        print(f"Refiner: {output['refiner_output'][:50]}...")
    
    # Find specific nodes
    print("\n=== Node Lookup ===")
    node_0_0_0 = tree.root.get_node_by_index([0, 0, 0])
    if node_0_0_0:
        print(f"Node [0,0,0]: {node_0_0_0.output[:50]}...")
    
    # Get all generator outputs
    print("\n=== Generator Analysis ===")
    generators = []
    def collect_generators(node):
        if node.type == "generator":
            generators.append(node)
        for child in node.children:
            collect_generators(child)
    
    collect_generators(tree.root)
    print(f"Found {len(generators)} generator nodes")
    for i, gen in enumerate(generators):
        print(f"  Generator {i}: {gen.output[:50]}...")
    
    # Analyze branching patterns
    print("\n=== Branching Analysis ===")
    def analyze_branching(node, depth=0):
        indent = "  " * depth
        print(f"{indent}{node.type}[{node.tree_index}]: {len(node.children)} children")
        for child in node.children:
            analyze_branching(child, depth + 1)
    
    analyze_branching(tree.root)


def demo_conversion():
    """Demonstrate converting from flat to tree format."""
    
    # Example flat data (current format)
    flat_data = [
        {
            'name': 'test_case',
            'tree_position': [0, 0, 0],
            'generator_output': 'Generator output 0',
            'verifier_output': 'Verifier output 0-0', 
            'refiner_output': 'Refiner output 0-0-0'
        },
        {
            'name': 'test_case',
            'tree_position': [0, 0, 1],
            'generator_output': 'Generator output 0',
            'verifier_output': 'Verifier output 0-0',
            'refiner_output': 'Refiner output 0-0-1'
        },
        {
            'name': 'test_case', 
            'tree_position': [0, 1, 0],
            'generator_output': 'Generator output 0',
            'verifier_output': 'Verifier output 0-1',
            'refiner_output': 'Refiner output 0-1-0'
        },
        {
            'name': 'test_case',
            'tree_position': [1, 0, 0],
            'generator_output': 'Generator output 1',
            'verifier_output': 'Verifier output 1-0',
            'refiner_output': 'Refiner output 1-0-0'
        }
    ]
    
    print("=== Converting Flat to Tree Format ===")
    
    # Build tree from flat data
    from tree_utils import build_tree_from_flat_data
    tree = build_tree_from_flat_data(flat_data)
    
    print(f"Converted to tree with {len(tree.get_all_leaves())} leaves")
    
    # Show the tree structure
    print("\nTree structure:")
    def print_tree(node, depth=0):
        indent = "  " * depth
        print(f"{indent}{node.type}[{node.tree_index}]: {node.output[:30]}...")
        for child in node.children:
            print_tree(child, depth + 1)
    
    print_tree(tree.root)
    
    # Convert back to flat format
    leaf_outputs = tree.get_leaf_outputs()
    print(f"\nConverted back to {len(leaf_outputs)} flat entries")
    for output in leaf_outputs:
        print(f"  {output['tree_index']}: {output['refiner_output']}")


if __name__ == "__main__":
    print("Tree Structure Demo")
    print("=" * 50)
    
    demo_tree_parsing()
    print("\n" + "=" * 50)
    demo_conversion()
