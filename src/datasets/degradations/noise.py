from src.datasets.degradations import BaseDegradation
from typing import Optional
from src.datasets.audio import AudioDataset
from src.utils import get_context
import torch
from torch import Tensor
from torchaudio.functional import add_noise


class AddNoise(BaseDegradation):
    """
    Helper class for adding noise to an audio tensor with a specified SNR range.
    The noise is sampled randomly from the given noise dataset.
    """

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
        self.deterministic = deterministic

    def _sample_snr(self, seed: Optional[int] = None) -> float:
        with get_context(seed, self.deterministic):
            return torch.empty(1).uniform_(self.min_snr, self.max_snr)

    def __call__(self, audio: Tensor, seed: Optional[int] = None) -> Tensor:
        noise = self.noise_dataset.sample()
        assert audio.shape == noise["waveform"].shape, "Audio and noise samples must have the same shape"
        snr = self._sample_snr(seed=seed)
        return add_noise(audio, noise["waveform"], snr=snr)
