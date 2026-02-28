from src.losses import MSELoss
from src import ModelOutput, Batch
import torch


def test_mse_loss():
    loss_fn = MSELoss()
    batch = Batch(input=torch.randn(16, 10), target=torch.randn(16, 1))
    model_output = ModelOutput(output=torch.randn(16, 1))
    loss = loss_fn.forward(model_output, batch)
    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"
