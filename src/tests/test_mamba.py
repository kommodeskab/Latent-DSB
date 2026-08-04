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


def test_mamba2_model_conditional():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping Mamba 2 test.")

    device = torch.device("cuda")

    # Instantiate conditional Mamba2Model on CUDA
    model = Mamba2Model(
        in_channels=1,
        out_channels=1,
        d_model=64,
        d_state=16,
        num_blocks=2,
        kernel_size=64,
        stride=4,
        conditional=True,
    ).to(device)

    batch_size = 4
    channels = 1
    length = 1024
    x = torch.randn(batch_size, channels, length, device=device, requires_grad=True)
    y = torch.randn(batch_size, channels, length, device=device, requires_grad=True)

    # Run forward pass with auxiliary waveform y
    out = model(x, y)

    assert out.shape == (
        batch_size,
        channels,
        length,
    ), f"Expected shape {(batch_size, channels, length)}, but got {out.shape}"

    # Assert gradients propagate to AdaLN parameters (and to y after weight update)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.all(x.grad == 0), "Gradients should propagate to x"

    # Due to AdaLN zero-initialization, d(out)/d(y) is initially 0 at step 0,
    # but the projection weights receive non-zero gradients to learn conditioning.
    adaln_weight = model.blocks[0].adaln.scale_proj.weight
    assert adaln_weight.grad is not None and not torch.all(
        adaln_weight.grad == 0
    ), "AdaLN scale_proj weights should receive non-zero gradients"

    # Verify that after an optimizer step, gradients flow directly to y
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    x.grad = None
    y.grad = None

    out = model(x, y)
    out.sum().backward()
    assert y.grad is not None and not torch.all(
        y.grad == 0
    ), "Gradients should propagate to y once AdaLN weights are non-zero"
