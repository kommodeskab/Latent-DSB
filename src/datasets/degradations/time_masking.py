from src.datasets.degradations.base import BaseDegradation
from torch import Tensor
import torch


class TimeMasking(BaseDegradation):
    """
    Helper class for applying time masking to an audio tensor.
    T
    """

    def __init__(self, max_mask_proportion: float, prob: float = 1.0, deterministic: bool = False):
        super().__init__(prob=prob, deterministic=deterministic)
        assert (
            0 <= max_mask_proportion <= 1
        ), "max_mask_proportion must be between 0 and 1 (representing the proportion of the audio to mask)"
        self.max_mask_proportion = max_mask_proportion
        self.deterministic = deterministic

    def fun(self, audio: Tensor) -> Tensor:
        C, T = audio.shape
        max_mask_length = int(T * self.max_mask_proportion)
        mask_length = torch.randint(0, max_mask_length + 1, (1,)).item()
        if mask_length == 0:
            return audio
        mask_start = torch.randint(0, T - mask_length + 1, (1,)).item()
        masked_audio = audio.clone()
        masked_audio[:, mask_start : mask_start + mask_length] = 0
        return masked_audio
