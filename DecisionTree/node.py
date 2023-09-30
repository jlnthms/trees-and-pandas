import pandas as pd
from DecisionTree.question import Question
from DecisionTree.utils import greedy_search
from Dataset.dataset import Dataset


class Node:
    def __init__(self, subset: Dataset):
        self.subset = subset
        self.question = None

    def select_feature(self):
        for feature in self.subset.data:
            if feature.dytpe == 'bool':
                self.question = Question(feature)
            else:
                threshold = greedy_search(self.subset, feature)
                self.question = Question(feature, threshold)


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
