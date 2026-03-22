from src.datasets.degradations import BaseDegradation
from typing import Optional
from torch import Tensor
from src.utils import get_context
import torch


class AddGaussianNoise(BaseDegradation):
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
