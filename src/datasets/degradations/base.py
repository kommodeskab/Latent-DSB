from typing import Optional
from torch import Tensor
from src.utils import get_context
import torch


class BaseDegradation:
    """
    Base class for audio degradations.

    Args:
        prob (float): Probability of applying degradation. Must be between 0 and 1.
        deterministic (bool): Whether to use deterministic behavior for reproducibility.
    """

    def __init__(self, prob: float = 0.0, deterministic: bool = False):
        assert 0 <= prob <= 1, "Probability must be between 0 and 1"
        self.prob = prob
        self.deterministic = deterministic

    def fun(self, audio: Tensor) -> Tensor:
        raise NotImplementedError("Degradation function not implemented")

    def __call__(self, audio: Tensor, seed: Optional[int] = None) -> Tensor:
        with get_context(seed, self.deterministic):
            if torch.rand(1).item() > self.prob:
                return audio

            return self.fun(audio)
