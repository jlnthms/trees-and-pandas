from DecisionTree.node import *


def termination_criteria_met(node: Node, depth, max_depth=None, min_samples=None):
    # 1. If the node is pure, create a leaf node with the majority class
    if node.subset.is_pure():
        return True
    # 2. If the depth limit is reached, create a leaf node with the majority class
    if max_depth is not None and depth >= max_depth:
        return True
    # 3. If the node has too few samples, create a leaf node with the majority class
    if min_samples is not None and len(node.subset) <= min_samples:
        return True
    # If none of the termination conditions are met, continue splitting
    return False


class DecisionTree:
    def __init__(self, dataset: Dataset):
        self.root = Node(dataset)

    def build(self, node, depth=0, max_depth=None, min_samples=None):
        if termination_criteria_met(node, depth, max_depth, min_samples):
            return Leaf(node.subset)  # Create a LeafNode with the majority class

        # Otherwise, select the best feature and threshold and create internal nodes
        node.select_feature()
        node.split_data()

        # Recursively build the true and false branches
        if node.true_child:
            self.build(node.true_child, depth + 1, max_depth, min_samples)
        if node.false_child:
            self.build(node.false_child, depth + 1, max_depth, min_samples)

    def fit(self, max_depth=None, min_samples=None):
        self.build(self.root, max_depth=max_depth, min_samples=min_samples)

    def print(self):
        pass
