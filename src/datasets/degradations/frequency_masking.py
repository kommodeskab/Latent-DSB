import torch
import torchaudio.transforms as T
from typing import Optional
from torch import Tensor
from src.utils import get_context
from src.datasets.degradations import BaseDegradation


class FrequencyMasking(BaseDegradation):
    def __init__(self, max_mask_length: int = 30, n_fft: int = 400, deterministic: bool = False):
        super().__init__()
        self.max_mask_length = max_mask_length
        self.n_fft = n_fft
        self.deterministic = deterministic

        self.to_spec = T.Spectrogram(n_fft=self.n_fft, power=None, center=True)
        self.from_spec = T.InverseSpectrogram(n_fft=self.n_fft, center=True)
        self.masker = T.FrequencyMasking(freq_mask_param=self.max_mask_length)

    def __call__(self, audio: Tensor, seed: Optional[int] = None) -> Tensor:
        context = get_context(seed=seed, deterministic=self.deterministic)

        with context:
            original_length = audio.shape[-1]

            complex_spec = self.to_spec.forward(audio)
            magnitude = complex_spec.abs()
            masked_magnitude = self.masker.forward(magnitude)

            binary_mask = (masked_magnitude != 0).to(complex_spec.dtype)

            masked_spec = complex_spec * binary_mask
            output_audio = self.from_spec.forward(masked_spec)

            if output_audio.shape[-1] > original_length:
                output_audio = output_audio[..., :original_length]
            elif output_audio.shape[-1] < original_length:
                padding = original_length - output_audio.shape[-1]
                output_audio = torch.nn.functional.pad(output_audio, (0, padding))

        return output_audio
