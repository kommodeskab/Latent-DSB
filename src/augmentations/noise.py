from src.augmentations.base import BaseAugmentation
from src.datasets import AudioDataset
from torchaudio.functional import add_noise
from src import AudioSample
from src.augmentations.resample import ResampleAugmentation
from src.augmentations.fit import CutAndFitAugmentation
import torch


class AddNoiseAugmentation(BaseAugmentation):
    def __init__(
        self,
        noise_dataset: AudioDataset,
        min_snr: float,
        max_snr: float,
        deterministic: bool = False,
    ):
        self.noise_dataset = noise_dataset
        self.min_snr = min_snr
        self.max_snr = max_snr
        assert self.min_snr <= self.max_snr, "min_snr must be less than max_snr"
        self.deterministic = deterministic  # TODO: implement deterministic behavior

    def __call__(self, sample: AudioSample) -> AudioSample:
        # randomly sample a noise sample from the noise dataset
        noise_sample: AudioSample = self.noise_dataset.sample()

        if not hasattr(self, "resample_aug"):
            self.resample_aug = ResampleAugmentation(new_sample_rate=sample["sample_rate"])
            self.fit_aug = CutAndFitAugmentation(
                target_length=sample["input"].shape[-1], deterministic=self.deterministic
            )

        noise_sample = self.resample_aug(noise_sample)
        noise_sample = self.fit_aug(noise_sample)

        # sample random uniform snr between min_snr and max_snr
        snr = torch.empty(1).uniform_(self.min_snr, self.max_snr)
        noisy_waveform = add_noise(sample["input"], noise_sample["input"], snr=snr)

        return AudioSample(
            input=noisy_waveform,
            sample_rate=sample["sample_rate"],
        )
