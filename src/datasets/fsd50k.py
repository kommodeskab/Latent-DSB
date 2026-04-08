from datasets import load_dataset
from src.datasets import BaseDataset
from typing import Literal
from src import AudioSample
import torch


class FSD50K(BaseDataset):
    """
    FSD50K (Free Sound Dataset 50K) is a large-scale noise dataset of human-labeled sound events.
    Available as a HuggingFace dataset: https://huggingface.co/datasets/philgzl/fsd50k
    """

    def __init__(self, split: Literal["train", "val"]):
        super().__init__()
        split = "dev" if split == "train" else "eval"  # naming convention from the HuggingFace dataset
        self.dataset = load_dataset("philgzl/fsd50k", cache_dir=self.data_path, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> AudioSample:
        sample = self.dataset[index]
        waveform = torch.from_numpy(sample["audio"]["array"]).unsqueeze(0).float()
        sample_rate = sample["audio"]["sampling_rate"]
        return AudioSample(waveform=waveform, sample_rate=sample_rate)


if __name__ == "__main__":
    dataset = FSD50K(split="train")
    input = dataset[0]
    print(input)
