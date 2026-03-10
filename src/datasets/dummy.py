import torch
from src.datasets import BaseDataset
from src import Sample


class DummyDataset(BaseDataset):
    """An example dataset to show how to implement a dataset."""

    def __init__(self, size: int = 1000):
        super().__init__()
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Sample:
        # this is totally random data so the model won't learn anything useful
        input = torch.randn(10)  # Example data: a random tensor of size 10
        target = torch.randn(1)  # Example target: a random tensor of size 1
        return Sample(input=input, target=target)


if __name__ == "__main__":
    dataset = DummyDataset(size=5)
    sample = dataset[0]
    print(sample)
