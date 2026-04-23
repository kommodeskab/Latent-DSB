from src.datasets.degradations import BaseDegradation
from src.datasets.audio import AudioDataset
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
        prob: float = 1.0,
        deterministic: bool = False,
    ):
        super().__init__(prob=prob, deterministic=deterministic)
        self.noise_dataset = noise_dataset
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.deterministic = deterministic

    def _sample_snr(self) -> Tensor:
        return torch.empty(1).uniform_(self.min_snr, self.max_snr)

    def fun(self, audio: Tensor) -> Tensor:
        noise = self.noise_dataset.sample()
        # if the noise sample is totally quiet, return the original audio
        if noise["waveform"].abs().max() < 1e-4:
            return audio

        assert audio.shape == noise["waveform"].shape, "Audio and noise samples must have the same shape"
        snr = self._sample_snr()
        noisy_audio = add_noise(audio, noise["waveform"], snr=snr)
        return noisy_audio
