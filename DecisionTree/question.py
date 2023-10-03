import pandas as pd

from Dataset.dataset import Dataset


class Question:
    def __init__(self, feature, threshold=None):
        self.feature = feature
        self.threshold = threshold

    def ask(self, example):
        if self.threshold:
            return example[self.feature] <= self.threshold
        return self.feature

    def answer(self, subset: Dataset):
        true_rows = []
        false_rows = []

        for i in subset.data.index:
            if self.ask(subset.data.iloc[i]):
                true_rows.append(subset.data.iloc[i])
            else:
                false_rows.append(subset.data.iloc[i])

        true_subset = pd.DataFrame(true_rows, columns=subset.data.columns).reset_index(drop=True)
        false_subset = pd.DataFrame(false_rows, columns=subset.data.columns).reset_index(drop=True)

        return Dataset(true_subset, subset.label), Dataset(false_subset, subset.label)

    def __str__(self):
        return f"Is {self.feature} <= {self.threshold}?"
