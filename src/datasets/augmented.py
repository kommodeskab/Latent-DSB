from src.augmentations.base import BaseAugmentation
from src.datasets import AudioDataset


class AugmentedAudioDataset(AudioDataset):
    def __init__(
        self,
        dataset: AudioDataset,
        augmentations: list[BaseAugmentation],
    ):
        self.dataset = dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        for aug in self.augmentations:
            sample = aug(sample)
        return sample
