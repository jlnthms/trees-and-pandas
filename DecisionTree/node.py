from DecisionTree.question import Question
from DecisionTree.utils import *
from Dataset.dataset import Dataset


class Node:
    def __init__(self, subset: Dataset, true_child=None, false_child=None):
        self.subset = subset
        self.question = None
        self.true_child = true_child
        self.false_child = false_child

    def select_feature(self):
        min_gini = float('inf')
        for feature in [col for col in self.subset.data.columns if col != self.subset.label]:
            if self.subset.data[feature].dtype == 'bool':
                candidate_question = Question(feature)
            else:
                threshold = greedy_search(self.subset, feature)
                candidate_question = Question(feature, threshold)
            true_set, false_set = candidate_question.answer(self.subset)
            impurity = gini_impurity(self.subset, true_set.data, false_set.data)
            if impurity < min_gini:
                min_gini, self.question = impurity, candidate_question

    def split_data(self):
        true_set, false_set = self.question.answer(self.subset)
        self.true_child, self.false_child = Node(true_set), Node(false_set)


class Leaf(Node):
    def __init__(self, subset: Dataset):
        super().__init__(subset)

    def predict(self):
        pass
