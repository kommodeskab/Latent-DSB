from datasets import load_dataset
from src.datasets import BaseDataset
from typing import Literal
from src import AudioSample
import torch
from torch.utils.data import ConcatDataset


class NewsReports01(BaseDataset):
    """
    Old news radio reports from WW2 (1940-1945).
    Taken from: https://archive.org/details/news01
    Available as a HuggingFace dataset: https://huggingface.co/datasets/Andemand11/news-reports-01
    """

    def __init__(self, split: Literal["train", "val"]):
        super().__init__()
        split = "train" if split == "train" else "test"  # naming convention from the HuggingFace dataset
        self.dataset = load_dataset("Andemand11/news-reports-01", cache_dir=self.data_path, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> AudioSample:
        sample = self.dataset[index]
        waveform = torch.from_numpy(sample["audio"]["array"]).unsqueeze(0).float()
        sample_rate = sample["audio"]["sampling_rate"]
        return AudioSample(waveform=waveform, sample_rate=sample_rate)


class NewsReports02(BaseDataset):
    """
    Old news radio reports from WW2 (1940-1945).
    Taken from: https://archive.org/details/news02
    Available as a HuggingFace dataset: https://huggingface.co/datasets/Andemand11/news-reports-02
    """

    def __init__(self, split: Literal["train", "val"]):
        super().__init__()
        split = "train" if split == "train" else "test"  # naming convention from the HuggingFace dataset
        self.dataset = load_dataset("Andemand11/news-reports-02", cache_dir=self.data_path, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> AudioSample:
        sample = self.dataset[index]
        waveform = torch.from_numpy(sample["audio"]["array"]).unsqueeze(0).float()
        sample_rate = sample["audio"]["sampling_rate"]
        return AudioSample(waveform=waveform, sample_rate=sample_rate)


class NewsReports(ConcatDataset):
    """
    Combined dataset of NewsReports01 and NewsReports02.
    """

    def __init__(self, split: Literal["train", "val"]):
        dataset1 = NewsReports01(split=split)
        dataset2 = NewsReports02(split=split)
        super().__init__([dataset1, dataset2])


if __name__ == "__main__":
    for split in ["train", "val"]:
        dataset = NewsReports(split=split)
        print(len(dataset))
        sample = dataset[0]
        print(sample)
