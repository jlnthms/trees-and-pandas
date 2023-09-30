import pandas as pd


class Dataset:
    def __init__(self, data: pd.DataFrame, label: str, true_label):
        self.data = data
        self.label = label
        self.true_label = true_label
