from src.datasets.degradations import BaseDegradation
from torch import Tensor
import torch


class AddGaussianNoise(BaseDegradation):
    def __init__(
        self,
        min_std: float,
        max_std: float,
        prob: float = 1.0,
        deterministic: bool = False,
    ):
        super().__init__(prob=prob, deterministic=deterministic)
        self.min_std = min_std
        self.max_std = max_std
        self.deterministic = deterministic

    def _sample_std(self) -> float:
        return torch.empty(1).uniform_(self.min_std, self.max_std)

    def fun(self, audio: Tensor) -> Tensor:
        std = self._sample_std()
        noise = torch.randn_like(audio) * std
        return audio + noise
