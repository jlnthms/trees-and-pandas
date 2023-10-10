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

    def sample_feature(self, frac):
        df = self.data.drop(self.label, axis=1)
        sampled_data = df.sample(frac=frac, axis='columns')
        sampled_data[self.label] = self.data[self.label]
        return Dataset(sampled_data, self.label)

    def bootstrap(self, frac):
        data = self.data.sample(frac=frac, axis='rows', replace=True).reset_index(drop=True)
        return Dataset(data, self.label)
