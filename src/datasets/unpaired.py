from src.datasets import BaseDataset
from src import UnpairedBatch


class UnpairedDataset(BaseDataset):
    """
    Given two datasets, return a random pair of samples.
    The datasets samples deterministcally form dataset0 and randomly from dataset1.
    The length of the dataset is defined as the length of dataset0, and dataset1 is sampled with replacement.
    """

    def __init__(
        self,
        dataset0: BaseDataset,
        dataset1: BaseDataset,
    ):
        super().__init__()
        self.dataset0 = dataset0
        self.dataset1 = dataset1

    def __len__(self) -> int:
        return len(self.dataset0)

    def __getitem__(self, idx: int) -> UnpairedBatch:
        x0 = self.dataset0[idx]
        x1 = self.dataset1.sample()

        return UnpairedBatch(
            x0=x0["input"],
            x1=x1["input"],
        )
