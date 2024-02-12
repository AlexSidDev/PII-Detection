import pandas as pd
import torch


class NERDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        tokens = torch.tensor(self.data['tokens'][index], dtype=torch.long)
        labels = torch.tensor(self.data['labels'][index], dtype=torch.long)
        return tokens, labels
