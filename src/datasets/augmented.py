from src.augmentations import BaseAugmentation
from src.datasets import AudioDataset


class AugmentedAudioDataset(AudioDataset):
    def __init__(
        self,
        base_dataset: AudioDataset,
        augmentations: list[BaseAugmentation],
    ):
        self.base_dataset = base_dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        for aug in self.augmentations:
            sample = aug(sample)
        return sample
