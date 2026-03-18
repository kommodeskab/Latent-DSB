from typing import Optional
from src.datasets.audio import AudioDataset
from src import DegradedAudioSample
import torch
from torchaudio.functional import add_noise
from src.utils import get_context
from torch import Tensor

class AddNoise:
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
    
class AddGaussianNoise:
    def __init__(
        self,
        min_std: float,
        max_std: float,
        deterministic: bool = False,
    ):
        self.min_std = min_std
        self.max_std = max_std
        self.deterministic = deterministic
        
    def _sample_std(self, seed: Optional[int] = None) -> float:
        with get_context(seed, self.deterministic):
            return torch.empty(1).uniform_(self.min_std, self.max_std)
        
    def __call__(self, audio: Tensor, seed: Optional[int] = None) -> Tensor:
        std = self._sample_std(seed=seed)
        noise = torch.randn_like(audio) * std
        return audio + noise

class TimeMasking:
    """
    Helper class for applying time masking to an audio tensor.
    The method will randomly sample a start index and a duration for each sample in the tensor.
    The masked region will be set to zero.
    """
    def __init__(self, max_mask_duration: int, deterministic: bool = False):
        self.max_mask_duration = max_mask_duration
        self.deterministic = deterministic
        
    def __call__(self, audio: Tensor, seed: Optional[int] = None) -> Tensor:
        with get_context(seed, self.deterministic):
            B, C, T = audio.shape
            masked_audio = audio.clone()
            for i in range(B):
                mask_duration = torch.randint(1, self.max_mask_duration + 1, (1,)).item()
                start_index = torch.randint(0, T - mask_duration + 1, (1,)).item()
                masked_audio[i, :, start_index:start_index + mask_duration] = 0
            return masked_audio

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
        self.add_noise = AddNoise(
            noise_dataset = noise_dataset,
            min_snr = min_snr,
            max_snr = max_snr,
            deterministic = deterministic,
        )

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx: int) -> DegradedAudioSample:
        clean = self.clean_dataset[idx]
        noisy_waveform = self.add_noise(clean["waveform"], seed=idx)

        return DegradedAudioSample(
            original_waveform=clean["waveform"],
            degraded_waveform=noisy_waveform,
            sample_rate=clean["sample_rate"],
        )


# TODO: implement a "ReverberantDegradedDataset" that adds reverb to the clean samples,
# using some kind of impulse response dataset
# good idea to find or upload a reverb dataset on HugginFace or similar

class VeryDegradedDataset(DegradedDataset):
    """
    A dataset that applies multiple degradations, including:
    - Additive Gaussian noise
    - Additive environmental noise
    - Reverberation
    - Clipping
    - Time masking
    - Frequency masking
    """
    
    def __init__(
        self,
        clean_dataset: AudioDataset,
        noise_dataset: AudioDataset,
        min_snr: float,
        max_snr: float,
        min_std: float,
        max_std: float,
        max_time_mask_duration: int,
        deterministic: bool = False,
    ):
        self.clean_dataset = clean_dataset
        self.noise_dataset = noise_dataset
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.max_time_mask_duration = max_time_mask_duration
        self.deterministic = deterministic
        
        self.add_noise = AddNoise(
            noise_dataset = noise_dataset,
            min_snr = min_snr,
            max_snr = max_snr,
            deterministic = deterministic,
        )
        self.time_masking = TimeMasking(
            max_mask_duration = max_time_mask_duration,
            deterministic = deterministic,
        )
        self.gaussian_noise = AddGaussianNoise(
            min_std = min_std,
            max_std = max_std,
            deterministic = deterministic,
        )
        
    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self, idx: int) -> DegradedAudioSample:
        clean = self.clean_dataset[idx]
        clean_waveform = clean["waveform"]
        
        noisy_waveform = self.add_noise(clean_waveform, seed=idx)
        noisy_waveform = self.time_masking(noisy_waveform, seed=idx)
        noisy_waveform = self.gaussian_noise(noisy_waveform, seed=idx)

        return DegradedAudioSample(
            original_waveform=clean_waveform,
            degraded_waveform=noisy_waveform,
            sample_rate=clean["sample_rate"],
        )