import pandas as pd

from DecisionTree.question import Question
from DecisionTree.node import Node, Leaf
from Dataset.dataset import Dataset


def termination_criteria_met(split_set: Dataset, depth, max_depth=None, min_samples=None):
    # 1. If the set is pure
    if split_set.is_pure():
        return True
    # 2. If the depth limit is reached, create a leaf node with the majority class
    if max_depth is not None and depth + 1 >= max_depth:
        return True
    # 3. If the node has too few samples, create a leaf node with the majority class
    if min_samples is not None and len(split_set) <= min_samples:
        return True
    # If none of the termination conditions are met, continue splitting
    return False


def greedy_search(subset: Dataset, feature: str):
    # return the threshold value that splits the set into the purest possible subsets
    lowest_gini, threshold = float('inf'), None
    for value in subset.data[feature].unique():
        true_set, false_set = Question(feature, value).answer(subset)
        gini = gini_impurity(subset, true_set.data, false_set.data)
        if gini < lowest_gini:
            lowest_gini, threshold = gini, value
    return threshold


def gini_impurity(parent_set: Dataset, true_set: pd.DataFrame, false_set: pd.DataFrame):
    classes = parent_set.data[parent_set.label].unique()

    true_set_size = len(true_set)
    false_set_size = len(false_set)
    total_set_size = len(parent_set)

    if false_set_size == 0 or true_set_size == 0:
        return 1.0

    true_class_probabilities = []
    false_class_probabilities = []

    for class_label in classes:
        true_class_count = len(true_set[true_set[parent_set.label] == class_label])
        true_class_probability = true_class_count / true_set_size
        true_class_probabilities.append(true_class_probability)

        false_class_count = len(false_set[false_set[parent_set.label] == class_label])
        false_class_probability = false_class_count / false_set_size
        false_class_probabilities.append(false_class_probability)

    true_set_impurity = 1 - sum(p ** 2 for p in true_class_probabilities)
    false_set_impurity = 1 - sum(p ** 2 for p in false_class_probabilities)

    weighted_impurity = (true_set_size / total_set_size) * true_set_impurity + \
                        (false_set_size / total_set_size) * false_set_impurity

    return weighted_impurity


def to_dot(node: Node):
    dot_data = "digraph DecisionTree {\n"
    dot_data += "node [shape=box, style=\"filled\", fillcolor=\"lightblue\"];\n"
    dot_data += _to_dot(node)
    dot_data += "}\n"
    return dot_data


def _to_dot(self, node):
    if isinstance(node, Leaf):
        label = f"Class: {node.predict()}"
    else:
        label = str(node.question)

    dot_data = f"\"{id(node)}\" [label=\"{label}\"];\n"

    if node.true_child:
        dot_data += f"\"{id(node)}\" -> \"{id(node.true_child)}\" [label=\"True\"];\n"
        dot_data += self._to_dot(node.true_child)

    if node.false_child:
        dot_data += f"\"{id(node)}\" -> \"{id(node.false_child)}\" [label=\"False\"];\n"
        dot_data += self._to_dot(node.false_child)

    return dot_data
