import pandas as pd

from DecisionTree.question import Question
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
    # Calculate the proportions of true and false samples in each subset
    true_set_size = len(true_set)
    false_set_size = len(false_set)

    total_set_size = true_set_size + false_set_size

    # Calculate Gini impurity for the true set
    if true_set_size == 0:
        true_set_impurity = 0  # Avoid division by zero
    else:
        true_set_true_prob = len(true_set[true_set[parent_set.label] == parent_set.true_label]) / true_set_size
        true_set_false_prob = 1 - true_set_true_prob
        true_set_impurity = 1 - (true_set_true_prob ** 2 + true_set_false_prob ** 2)

    # Calculate Gini impurity for the false set
    if false_set_size == 0:
        false_set_impurity = 0  # Avoid division by zero
    else:
        false_set_true_prob = len(false_set[false_set[parent_set.label] == parent_set.true_label]) / false_set_size
        false_set_false_prob = 1 - false_set_true_prob
        false_set_impurity = 1 - (false_set_true_prob ** 2 + false_set_false_prob ** 2)

    # Calculate the weighted average Gini impurity
    weighted_impurity = (true_set_size / total_set_size) * true_set_impurity + \
                        (false_set_size / total_set_size) * false_set_impurity

    return weighted_impurity
