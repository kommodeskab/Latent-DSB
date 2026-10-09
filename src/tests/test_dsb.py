import torch
import torch.nn as nn
from src import UnpairedAudioBatch, StepOutput
from src.lightning_modules.dsb import DSB
from src.lightning_modules.scheduler import DSBScheduler
from src.networks.encoders import IdentityEncoderDecoder
from src.losses import MSELoss


class MockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_input_shapes = {}

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor, conditional: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.last_input_shapes = {
            "xt": xt.shape,
            "timesteps": timesteps.shape,
            "conditional": conditional.shape,
        }
        # Return an output with the same shape as xt
        return xt * 2.0


def test_dsb_common_step_batch_stacking():
    """
    Test that DSB.common_step stacks backward and forward batches into a single batch
    before passing to the model forward pass, then splits outputs to compute backward
    and forward losses correctly.
    """
    batch_size = 4
    dim = 16
    x0 = torch.randn(batch_size, dim)
    x1 = torch.randn(batch_size, dim)
    batch = UnpairedAudioBatch(
        x0=x0,
        x1=x1,
        x1_clean=None,
        sample_rate=[16000] * batch_size,
    )

    mock_net = MockNetwork()
    encoder_decoder = IdentityEncoderDecoder()
    scheduler = DSBScheduler(epsilon=1.0, target="terminal", condition_on_start=False)
    loss_fn = MSELoss()

    dsb = DSB(
        model=mock_net,
        encoder_decoder=encoder_decoder,
        pretraining_steps=100,
        inference_steps=5,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )

    step_output: StepOutput = dsb.common_step(batch, batch_idx=0)

    # Verify that model was called exactly once with stacked batch of size 2 * batch_size
    assert mock_net.call_count == 1
    assert mock_net.last_input_shapes["xt"][0] == 2 * batch_size
    assert mock_net.last_input_shapes["timesteps"][0] == 2 * batch_size
    assert mock_net.last_input_shapes["conditional"][0] == 2 * batch_size

    # Verify step_output structure
    assert "loss" in step_output
    assert step_output["model_output"]["output"].shape[0] == 2 * batch_size
    loss_output = step_output["loss_output"]
    assert "backward_loss" in loss_output
    assert "forward_loss" in loss_output
    expected_mean_loss = (loss_output["backward_loss"] + loss_output["forward_loss"]) / 2.0
    assert torch.isclose(step_output["loss"], expected_mean_loss)
