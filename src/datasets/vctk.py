from datasets import load_dataset
from src.datasets import BaseDataset
from typing import Literal
from src import AudioSample
import torch

import os

os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"


class VCTK(BaseDataset):
    def __init__(self, split: Literal["train", "val"]):
        super().__init__()
        split = "validation" if split == "val" else split  # naming convention from the HuggingFace dataset
        self.split = split
        self.dataset = load_dataset("badayvedat/VCTK", split=split, cache_dir=self.data_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> AudioSample:
        sample = self.dataset[index]
        input = torch.from_numpy(sample["flac"]["array"]).unsqueeze(0).float()
        sample_rate = sample["flac"]["sampling_rate"]
        return AudioSample(waveform=input, sample_rate=sample_rate)


if __name__ == "__main__":
    dataset = VCTK(split="train")
    input = dataset[0]["waveform"]
    print(input.shape)
    print(type(input))
