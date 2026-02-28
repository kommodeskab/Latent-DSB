from datasets import load_dataset
from src.datasets import BaseDataset
from typing import Literal
from src import AudioSample
import torch


class WHAM(BaseDataset):
    def __init__(self, split: Literal["train", "val", "test"]):
        super().__init__()
        split = "validation" if split == "val" else split  # naming convention from the HuggingFace dataset
        self.split = split
        self.dataset = load_dataset("philgzl/wham", split=split, cache_dir=self.data_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> AudioSample:
        sample = self.dataset[index]
        input = torch.from_numpy(sample["audio"]["array"]).unsqueeze(0).float()
        sample_rate = sample["audio"]["sampling_rate"]
        return AudioSample(input=input, sample_rate=sample_rate)


if __name__ == "__main__":
    dataset = WHAM("train")
    input = dataset[0]["input"]
    print(input.shape)
    print(type(input))
