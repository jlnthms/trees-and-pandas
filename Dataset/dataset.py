import pandas as pd


class Dataset:
    def __init__(self, data: pd.DataFrame, label: str):
        self.data = data
        self.label = label

    def __len__(self):
        # number of rows
        return len(self.data.index)

    def is_pure(self):
        unique_labels = self.data[self.label].unique()
        return len(unique_labels) == 1
