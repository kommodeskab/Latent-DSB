from typing import Optional
from src.datasets.audio import AudioDataset
from src import DegradedAudioSample
import torch
from torchaudio.functional import add_noise
from src.utils import temporary_seed
from contextlib import nullcontext


class DegradedDataset(AudioDataset):
    """
    Given a dataset, this dataset applies some kind of degradation to the audio samples.
    __getitem__ then returns a "DegradedAudioSample" which has both the original and degraded audio, as well as the sample rate.
    """

    def __getitem__(self, idx: int) -> DegradedAudioSample: ...


class NoisyDegradedDataset(DegradedDataset):
    def __init__(
        self,
        clean_dataset: AudioDataset,
        noise_dataset: AudioDataset,
        min_snr: float,
        max_snr: float,
        deterministic: bool = False,
    ):
        self.clean_dataset = clean_dataset
        self.noise_dataset = noise_dataset
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.deterministic = deterministic

    def __len__(self):
        return len(self.clean_dataset)

    def _sample_snr(self, seed: Optional[int] = None) -> float:
        context = temporary_seed(seed) if self.deterministic else nullcontext()
        with context:
            return torch.empty(1).uniform_(self.min_snr, self.max_snr)

    def __getitem__(self, idx: int) -> DegradedAudioSample:
        clean = self.clean_dataset[idx]
        noise = self.noise_dataset.sample()

        assert clean["sample_rate"] == noise["sample_rate"], "Clean and noise samples must have the same sample rate"
        assert clean["waveform"].shape == noise["waveform"].shape, "Clean and noise samples must have the same shape"

        snr = self._sample_snr(seed=idx)
        noisy_waveform = add_noise(clean["waveform"], noise["waveform"], snr=snr)

        return DegradedAudioSample(
            original_waveform=clean["waveform"],
            degraded_waveform=noisy_waveform,
            sample_rate=clean["sample_rate"],
        )
