import pandas as pd

from DecisionTree.tree import DecisionTree
from Dataset.dataset import *


class RandomForest:
    def __init__(self, dataset: Dataset, size):
        self.dataset = dataset
        self.size = size
        self.trees = []

    def fit(self, sampling_frac=0.33, bagging_frac=0.8):
        for _ in range(self.size):
            root_dataset = self.dataset.sample_feature(sampling_frac).bootstrap(bagging_frac)
            self.trees.append(DecisionTree(root_dataset))

    def predict(self, X: pd.DataFrame):
        predictions = pd.DataFrame()
        for tree in self.trees:
            predictions = pd.concat([predictions, tree.predict(X)], axis=1)

        vote = predictions.mode(axis=1).iloc[:, 0]
        return pd.DataFrame({'Vote': vote})



