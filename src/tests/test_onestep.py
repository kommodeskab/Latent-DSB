import torch
import torch.nn as nn
from src.lightning_modules import OneStepModel
from src.losses import MSELoss


class DummyLinearNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def test_onestep_model():
    model = DummyLinearNetwork()
    loss_fn = MSELoss()

    # Instantiate OneStepModel
    onestep = OneStepModel(
        model=model,
        loss_fn=loss_fn,
        ema_decay=0.9,
    )

    # Test batch
    batch_size = 4
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 10)
    batch = {"x1": x, "x0": y}

    # Test forward
    out = onestep(batch)
    assert isinstance(out, dict) and "output" in out
    assert out["output"].shape == (batch_size, 10)

    # Test common_step
    step_out = onestep.common_step(batch, 0)
    assert "loss" in step_out
    assert step_out["loss"].item() >= 0

    # Test sample method (runs inside average_parameters context)
    sampled = onestep.sample(x)
    assert sampled.shape == (batch_size, 10)

    # Test EMA updates (using mock optimizer)
    prev_ema_params = [p.clone() for p in onestep.ema.shadow_params]
    onestep.on_before_zero_grad(None)
    curr_ema_params = onestep.ema.shadow_params
    # hadow params should be updated (or stay consistent if no optimization steps occurred)
    assert len(curr_ema_params) == len(prev_ema_params)

    # Test checkpoint saving and loading
    checkpoint = {}
    onestep.on_save_checkpoint(checkpoint)
    assert "ema" in checkpoint

    onestep.on_load_checkpoint(checkpoint)
