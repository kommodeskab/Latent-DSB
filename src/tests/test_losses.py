from src.losses import MSELoss
from src import ModelOutput, Batch
import torch


def test_mse_loss():
    loss_fn = MSELoss()
    batch = Batch(input=torch.randn(16, 10), target=torch.randn(16, 1))
    model_output = ModelOutput(output=torch.randn(16, 1))
    loss = loss_fn(model_output, batch)
    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"


def test_drifting_loss():
    from src.losses import DriftingLoss

    loss_fn = DriftingLoss(temperature=1.0)
    batch = Batch(input=torch.randn(4, 1, 16000), target=torch.randn(4, 1, 16000))
    model_output = ModelOutput(output=torch.randn(4, 1, 16000))
    model_output["output"].requires_grad = True

    loss = loss_fn(model_output, batch)

    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"

    # Check that gradient propagation works
    loss["loss"].backward()
    assert model_output["output"].grad is not None, "Gradients should propagate to model output"
    assert not torch.all(model_output["output"].grad == 0), "Gradients should be non-zero"


def test_drifting_loss_dynamic():
    from src.losses import DriftingLoss

    loss_fn = DriftingLoss(temperature="dynamic")
    batch = Batch(input=torch.randn(4, 1, 16000), target=torch.randn(4, 1, 16000))
    model_output = ModelOutput(output=torch.randn(4, 1, 16000))
    model_output["output"].requires_grad = True

    loss = loss_fn(model_output, batch)

    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"

    # Check that gradient propagation works
    loss["loss"].backward()
    assert model_output["output"].grad is not None, "Gradients should propagate to model output"
    assert not torch.all(model_output["output"].grad == 0), "Gradients should be non-zero"
