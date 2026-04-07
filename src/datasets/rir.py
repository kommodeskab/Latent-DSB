from datasets import load_dataset, Audio
from src.datasets import BaseDataset
from typing import Literal
from src import AudioSample
import torch

import os

os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"


class RIR(BaseDataset):
    def __init__(self, split: Literal["train", "test", "val"], mono=True):
        super().__init__()
        match split:
            case "train":
                split = "train"
            case "test":
                split = "test"
            case "val":
                split = "validation"

        # Note that mono returns rirs with just one channel - binaural has general multi-channel rirs., most of them 2, some 4, some 16, idk.
        self.dataset = load_dataset(
            path="andnymand/RIR-datasets",
            name="mono" if mono else "binaural",
            split=split,
            cache_dir=self.data_path,
        )

        # Always say mono = false here - hugging face loads lazily, so it does not slow mono files down, and skips an unnecessary conditional
        # Also makes troubleshooting more clear later perhaps
        self.dataset = self.dataset.cast_column("audio", Audio(mono=False))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> AudioSample:
        sample = self.dataset[index]
        input = torch.from_numpy(sample["audio"]["array"])
        if input.ndim == 1:
            input = input.unsqueeze(0)
        sample_rate = sample["audio"]["sampling_rate"]
        return AudioSample(waveform=input, sample_rate=sample_rate)


if __name__ == "__main__":
    dataset_train = RIR(split="train")
    dataset_test = RIR(split="test")
    dataset_val = RIR(split="val")
    dataset_bin = RIR(split="train", mono=False)
    train = dataset_train[0]["waveform"]
    test = dataset_test[0]["waveform"]
    val = dataset_val[0]["waveform"]
    bin = dataset_bin[0]["waveform"]
    print(train.shape)
    print(test.shape)
    print(val.shape)
    print(bin.shape)
    print(bin[:, 0:6])
    print(type(train))
