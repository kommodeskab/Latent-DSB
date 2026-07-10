import torch
import pytest
from src.networks import Mamba2Model


def test_mamba2_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping Mamba 2 test.")

    device = torch.device("cuda")

    # Instantiate Mamba2Model on CUDA
    model = Mamba2Model(
        in_channels=1,
        out_channels=1,
        d_model=64,
        d_state=16,
        num_blocks=2,
        kernel_size=64,
        stride=4,
    ).to(device)

    # Create dummy inputs: batch_size=4, channels=1, length=1024 on CUDA
    batch_size = 4
    channels = 1
    length = 1024
    x = torch.randn(batch_size, channels, length, device=device, requires_grad=True)

    # Run forward pass (should not need any timestep or class embeddings)
    out = model(x)

    # Assert output shape matches input shape
    assert out.shape == (
        batch_size,
        channels,
        length,
    ), f"Expected shape {(batch_size, channels, length)}, but got {out.shape}"

    # Assert gradients propagate back to x
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should propagate to the input tensor"
    assert not torch.all(x.grad == 0), "Gradients should be non-zero"
