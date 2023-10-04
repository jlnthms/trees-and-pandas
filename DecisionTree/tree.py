import pandas as pd

from DecisionTree.node import *


class DecisionTree:
    def __init__(self, dataset: Dataset):
        self.root = Node(dataset)

    def build(self, node, depth=0, max_depth=None, min_samples=None):
        if isinstance(node, Leaf):
            return

        # Otherwise, select the best feature and threshold and create internal nodes
        node.select_feature()
        node.split_data(depth, max_depth, min_samples)

        # Recursively build the true and false branches
        if node.true_child:
            self.build(node.true_child, depth + 1, max_depth, min_samples)
        if node.false_child:
            self.build(node.false_child, depth + 1, max_depth, min_samples)

    def fit(self, max_depth=None, min_samples=None):
        self.build(self.root, max_depth=max_depth, min_samples=min_samples)

    def predict(self, X: pd.DataFrame):
        prediction_labels = []
        for _, row in X.iterrows():
            node = self.root
            while not isinstance(node, Leaf):
                if node.question.ask(row) == True:
                    node = node.true_child
                else:
                    node = node.false_child
            prediction_labels.append(node.predict())
        return pd.DataFrame(prediction_labels)

    def print(self, node=None, indent=""):
        if node is None:
            node = self.root

        # If it's a leaf node, print the class or value
        if isinstance(node, Leaf):
            print(f"{indent}|--- class: {node.predict()}")
            return

        # If it's an internal node, print the question and recurse
        print(f"{indent}|--- {node.question}")
        print(f"{indent}|   |--- True:")
        self.print(node.true_child, indent + "|   ")
        print(f"{indent}|   |--- False:")
        self.print(node.false_child, indent + "|   ")
