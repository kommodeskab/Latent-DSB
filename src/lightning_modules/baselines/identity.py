from src.lightning_modules import BaseLightningModule
from torch import Tensor


class Identity(BaseLightningModule):
    """
    A simple identity module that returns the input as output. Useful for testing and as a baseline.
    Used as a baseline
    """

    def __init__(self):
        super().__init__()

    def common_step(self, batch, batch_idx):
        return ...

    def sample(self, x_start: Tensor, **kwargs) -> Tensor:
        return x_start
