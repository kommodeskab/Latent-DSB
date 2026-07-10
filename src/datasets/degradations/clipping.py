from src.datasets.degradations import BaseDegradation
import torch
from torch import Tensor, clamp


class Clip(BaseDegradation):
    """
    Helper class for clipping an audio clip. Removes power between
    min_db and max_db from the given sample.
    """

    def __init__(
        self,
        min_db: float,
        max_db: float,
        binary_search_iterations: int = 25,
    ):
        self.min_db = min_db
        self.max_db = max_db
        self.binary_search_iterations = binary_search_iterations

    def _sample_db(self) -> Tensor:
        return torch.empty(1).uniform_(self.min_db, self.max_db)

    def binary_clamp_search(self, audio: Tensor, target_db_loss: float):
        x = audio
        target_p = torch.mean(x**2) * (10 ** (-target_db_loss / 10))
        low = 0.0
        high = torch.max(torch.abs(x))

        for _ in range(self.binary_search_iterations):
            mid = (low + high) / 2
            # Check power at this threshold
            p = torch.mean(clamp(x, -mid, mid) ** 2)
            if p < target_p:
                low = mid
            else:
                high = mid
        mid = (low + high) / 2

        return clamp(x, -mid, mid)

    def fun(self, audio: Tensor) -> Tensor:
        db = self._sample_db()
        return self.binary_clamp_search(audio, db.item())


if __name__ == "__main__":
    x = torch.rand((1, 48000))
    clip = Clip(5.0, 10.0, 1.0)
    x_clipped = clip(x)
    print(
        f"Power reduction of clipped x - \
        {10 * torch.log10(torch.mean(x**2)) - 10 * torch.log10(torch.mean(x_clipped**2))}"
    )
