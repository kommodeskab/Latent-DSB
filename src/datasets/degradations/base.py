from typing import Optional
from torch import Tensor


class BaseDegradation:
    def __call__(self, audio: Tensor, seed: Optional[int] = None) -> Tensor: ...
