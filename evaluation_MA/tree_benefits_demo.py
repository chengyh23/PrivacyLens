#!/usr/bin/env python3
"""
Demo script showing the benefits of tree structure over flat format.
"""

import json
from tree_utils import MultiAgentTree, TreeNode, build_tree_from_flat_data

def compare_formats():
    """Compare flat vs tree format for memory usage and analysis capabilities."""
    
    # Simulate data for branching factor 3 (3x3x3 = 27 leaves)
    n_branching = 3
    flat_data = []
    
    print("=== Generating Sample Data ===")
    print(f"Branching factor: {n_branching}")
    print(f"Total leaves: {n_branching**3}")
    
    # Generate flat data (current format)
    for i in range(n_branching):
        gen_output = f"Generator output {i}"
        for j in range(n_branching):
            ver_output = f"Verifier output {i}-{j}"
            for k in range(n_branching):
                ref_output = f"Refiner output {i}-{j}-{k}"
                flat_data.append({
                    'name': 'test_case',
                    'tree_position': [i, j, k],
                    'generator_output': gen_output,
                    'verifier_output': ver_output,
                    'refiner_output': ref_output
                })
    
    print(f"Flat format: {len(flat_data)} entries")
    
    # Convert to tree format
    tree = build_tree_from_flat_data(flat_data)
    
    print(f"Tree format: 1 tree with {len(tree.get_all_leaves())} leaves")
    
    # Memory usage comparison
    flat_json = json.dumps(flat_data)
    tree_json = json.dumps(tree.to_dict())
    
    print(f"\n=== Memory Usage ===")
    print(f"Flat format size: {len(flat_json)} characters")
    print(f"Tree format size: {len(tree_json)} characters")
    print(f"Compression ratio: {len(flat_json) / len(tree_json):.2f}x")
    
    # Analysis capabilities
    print(f"\n=== Analysis Capabilities ===")
    
    # 1. Find all generator outputs (unique)
    generators = []
    def collect_generators(node):
        if node.type == "generator":
            generators.append(node.output)
        for child in node.children:
            collect_generators(child)
    
    collect_generators(tree.root)
    print(f"Unique generator outputs: {len(set(generators))}")
    
    # 2. Find all paths through the tree
    all_paths = []
    for leaf in tree.get_all_leaves():
        path = leaf.get_path_to_root()
        all_paths.append([node.tree_index for node in path])
    
    print(f"Total paths: {len(all_paths)}")
    
    # 3. Analyze decision points
    print(f"\n=== Decision Point Analysis ===")
    
    # Count nodes at each level
    level_counts = {}
    def count_levels(node, level=0):
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
        for child in node.children:
            count_levels(child, level + 1)
    
    count_levels(tree.root)
    for level, count in sorted(level_counts.items()):
        node_type = ["root", "generator", "verifier", "refiner"][level]
        print(f"Level {level} ({node_type}): {count} nodes")
    
    # 4. Find alternative paths to same outcome
    print(f"\n=== Path Analysis ===")
    
    # Group leaves by their final output
    output_groups = {}
    for leaf in tree.get_all_leaves():
        output = leaf.output
        if output not in output_groups:
            output_groups[output] = []
        output_groups[output].append(leaf.tree_index)
    
    print(f"Unique final outputs: {len(output_groups)}")
    for output, indices in list(output_groups.items())[:3]:  # Show first 3
        print(f"  Output '{output[:30]}...' appears in {len(indices)} paths: {indices}")
    
    # 5. Tree traversal examples
    print(f"\n=== Tree Traversal Examples ===")
    
    # Find all verifier nodes that approved (contain "good" or "approve")
    approved_verifiers = []
    def find_approved_verifiers(node):
        if node.type == "verifier" and ("good" in node.output.lower() or "approve" in node.output.lower()):
            approved_verifiers.append(node.tree_index)
        for child in node.children:
            find_approved_verifiers(child)
    
    find_approved_verifiers(tree.root)
    print(f"Approved verifiers: {approved_verifiers}")
    
    # Find all paths that go through a specific generator
    generator_0_paths = []
    for leaf in tree.get_all_leaves():
        path = leaf.get_path_to_root()
        if len(path) > 1 and path[1].tree_index[0] == 0:  # Generator 0
            generator_0_paths.append(leaf.tree_index)
    
    print(f"Paths through generator 0: {generator_0_paths}")
    
    # 6. Export specific subtrees
    print(f"\n=== Subtree Analysis ===")
    
    # Get all paths that start with generator 0
    gen_0_node = tree.root.get_node_by_index([0])
    if gen_0_node:
        gen_0_subtree = MultiAgentTree("gen_0_subtree", gen_0_node)
        gen_0_leaves = gen_0_subtree.get_all_leaves()
        print(f"Generator 0 subtree has {len(gen_0_leaves)} leaves")
        
        # Show decision tree for generator 0
        print("Generator 0 decision tree:")
        def print_decision_tree(node, depth=0):
            indent = "  " * depth
            print(f"{indent}{node.type}[{node.tree_index}]")
            for child in node.children:
                print_decision_tree(child, depth + 1)
        
        print_decision_tree(gen_0_node)


def demonstrate_tree_operations():
    """Demonstrate various tree operations that are easy with tree structure."""
    
    print("\n" + "=" * 60)
    print("TREE OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Load example tree
    with open('example_tree_structure.json', 'r') as f:
        tree_data = json.load(f)
    
    tree = MultiAgentTree.from_dict(tree_data)
    
    # 1. Find all paths that lead to privacy-conscious decisions
    privacy_paths = []
    for leaf in tree.get_all_leaves():
        if "privacy" in leaf.output.lower():
            path = leaf.get_path_to_root()
            privacy_paths.append([node.tree_index for node in path])
    
    print(f"Privacy-conscious paths: {privacy_paths}")
    
    # 2. Compare different generator strategies
    generator_strategies = {}
    for leaf in tree.get_all_leaves():
        path = leaf.get_path_to_root()
        if len(path) > 1:
            gen_idx = path[1].tree_index[0]
            if gen_idx not in generator_strategies:
                generator_strategies[gen_idx] = []
            generator_strategies[gen_idx].append(leaf.output)
    
    print(f"\nGenerator strategies:")
    for gen_idx, outputs in generator_strategies.items():
        print(f"  Generator {gen_idx}: {len(outputs)} outcomes")
        for output in outputs[:2]:  # Show first 2
            print(f"    - {output[:50]}...")
    
    # 3. Find consensus paths (where verifier and refiner agree)
    consensus_paths = []
    for leaf in tree.get_all_leaves():
        path = leaf.get_path_to_root()
        if len(path) >= 3:
            verifier_output = path[2].output
            refiner_output = path[3].output
            # Simple consensus check (both mention similar concepts)
            if ("privacy" in verifier_output.lower() and "privacy" in refiner_output.lower()) or \
               ("approve" in verifier_output.lower() and "proceed" in refiner_output.lower()):
                consensus_paths.append(leaf.tree_index)
    
    print(f"\nConsensus paths: {consensus_paths}")
    
    # 4. Extract decision patterns
    print(f"\nDecision patterns:")
    
    # Pattern: Generator -> Verifier approval -> Refiner implementation
    approval_patterns = []
    for leaf in tree.get_all_leaves():
        path = leaf.get_path_to_root()
        if len(path) >= 3:
            verifier_output = path[2].output
            if "approve" in verifier_output.lower() or "good" in verifier_output.lower():
                approval_patterns.append(leaf.tree_index)
    
    print(f"  Approval patterns: {approval_patterns}")
    
    # Pattern: Generator -> Verifier concerns -> Refiner fixes
    concern_patterns = []
    for leaf in tree.get_all_leaves():
        path = leaf.get_path_to_root()
        if len(path) >= 3:
            verifier_output = path[2].output
            if "concern" in verifier_output.lower() or "issue" in verifier_output.lower():
                concern_patterns.append(leaf.tree_index)
    
    print(f"  Concern patterns: {concern_patterns}")


if __name__ == "__main__":
    compare_formats()
    demonstrate_tree_operations()
