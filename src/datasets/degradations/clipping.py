from sympy.physics.mechanics import vector
from src.datasets.degradations import BaseDegradation
from src.datasets.audio import AudioDataset
import torch
from torch import Tensor, clamp


class Clip(BaseDegradation):
    """
    Helper class for clipping an audio clip. Removes power between
    min_dB and max_dB from the given sample.
    """

    def __init__(
        self,
        min_dB: float,
        max_dB: float,
        prob: float = 0.0,
        deterministic: bool = False,
        binary_search_iterations: int = 25
    ):
        super().__init__(prob=prob, deterministic=deterministic)
        self.min_dB = min_dB
        self.max_dB = max_dB
        self.deterministic = deterministic
        self.binary_search_iterations = binary_search_iterations

    def _sample_dB(self) -> Tensor:
        return torch.empty(1).uniform_(self.min_dB, self.max_dB)

    def binary_clamp_search(self, audio: Tensor, target_db_loss: float):
        x= audio
        target_p = torch.mean(x**2) * (10**(-target_db_loss / 10))
        low = 0.0
        high = torch.max(torch.abs(x))
        
        for _ in range(self.binary_search_iterations):
            mid = (low + high) / 2
            # Check power at this threshold
            p = torch.mean(clamp(x, -mid, mid)**2)
            if p < target_p:
                low = mid
            else:
                high = mid
        mid = (low + high) / 2
                
        return clamp(x, -mid, mid)

    def fun(self, audio: Tensor) -> Tensor:
        dB = self._sample_dB()
        return self.binary_clamp_search(audio,dB.item())

if __name__ == "__main__":
    x = torch.rand((1,48000))
    clip = Clip(5.0,10.0,1.0)
    x_clipped = clip(x)
    print(f'Power reduction of clipped x - \
        {10*torch.log10(torch.mean(x**2))-10*torch.log10(torch.mean(x_clipped**2))}')
