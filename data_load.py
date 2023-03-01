import pandas as pd
from torch.utils.data import Dataset


def load_data(path):
    # 加载数据
    data = pd.read_csv(path, index_col=None)
    # 数据标题
    label = data['attack_types']
    data.drop('attack_types', axis=1, inplace=True)
    return data.values, label


class MyDataSet(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.dataset, self.label = load_data(path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        label = self.label[idx]
        if self.transform:
            dataset = self.transform(dataset)
        if self.target_transform:
            label = self.target_transform(label)
        return dataset, label
