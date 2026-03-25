from datasets import load_dataset
from src.datasets import BaseDataset
import torch
from typing import Literal
from src import AudioSample


class LibriSpeech(BaseDataset):
    def __init__(self, split: Literal["train", "val", "test"]):
        super().__init__()
        match split:
            case "train":
                split = "train.100"
            case "val":
                split = "validation"
            case "test":
                split = "test-clean"

        self.dataset = load_dataset("openslr/librispeech_asr", "clean", split=split, cache_dir=self.data_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        input = torch.from_numpy(sample["audio"]["array"]).unsqueeze(0).float()
        sample_rate = sample["audio"]["sampling_rate"]
        return AudioSample(waveform=input, sample_rate=sample_rate)
