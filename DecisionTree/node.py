from DecisionTree.question import Question
from DecisionTree.utils import *
from Dataset.dataset import Dataset


class Node:
    def __init__(self, subset: Dataset):
        self.subset = subset
        self.question = None

    def select_feature(self):
        for feature in [col for col in self.subset.data.columns if col != self.subset.label]:
            min_gini = float('inf')
            if self.subset.data[feature].dtype == 'bool':
                candidate_question = Question(feature)
            else:
                threshold = greedy_search(self.subset, feature)
                candidate_question = Question(feature, threshold)
            true_set, false_set = candidate_question.answer(self.subset)
            impurity = gini_impurity(self.subset, true_set, false_set)
            if impurity < min_gini:
                min_gini, self.question = impurity, candidate_question


class Root(Node):
    def __init__(self, subset: Dataset):
        super().__init__(subset)
        self.left_child = None
        self.right_child = None


class Internal(Node):
    def __init__(self, subset: Dataset):
        super().__init__(subset)
        self.parent = None
        self.left_child = None
        self.right_child = None


class Leaf(Node):
    def __init__(self, subset: Dataset):
        super().__init__(subset)
        self.parent = None
