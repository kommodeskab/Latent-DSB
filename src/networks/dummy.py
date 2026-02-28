import torch.nn as nn
from torch import Tensor
import torch


class DummyNetwork(nn.Module):
    """A very simply linear network for demonstration purposes."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


if __name__ == "__main__":
    model = DummyNetwork(input_size=10, output_size=2)
    sample_input = torch.randn(1, 10)
    output = model(sample_input)
    print(output)
