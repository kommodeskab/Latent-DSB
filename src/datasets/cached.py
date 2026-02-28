from src.datasets import BaseDataset


class CachedDataset(BaseDataset):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset
        print(f"Loading {len(dataset)} samples into memory...")
        self.cache = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx: int):
        return self.cache[idx]


class LazyCachedDataset(BaseDataset):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]
