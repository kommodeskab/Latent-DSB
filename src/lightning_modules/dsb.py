from torch import Tensor
import torch
from src import UnpairedAudioBatch, ModelOutput, StepOutput, SchedulerBatch
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from functools import partial
from src.networks.encoders import BaseEncoderDecoder
from typing import Optional
from src.losses import BaseLossFunction
from src.lightning_modules.scheduler import DSBScheduler, DIRECTIONS, SCHEDULER_TYPES
from torch_ema import ExponentialMovingAverage
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft


class DSB(BaseLightningModule):
    """
    A Latent Diffusion Schrödinger Bridge (DSB) model for unpaired audio translation.

    Args:
        model (Module): The main network. Should accept as input a tensor x, a tensor of timesteps t, and a tensor of conditionals c.
        encoder_decoder (BaseEncoderDecoder): An encoder-decoder network for encoding and decoding audio samples to and from the latent space.
        pretraining_steps (int): The number of training step before switching to "finetuning" mode where the model generates it own training examples.
        inference_steps (int): The number of steps to use during inference (sampling). Higher = better quality but slower training. Usually around 5 is satisfactory
        scheduler (DSBScheduler): The DSB scheduler that defines the noise schedule and the sampling procedure. See the scheduler class for more details.
        loss_fn (Optional[BaseLossFunction]): The loss function to use during training.
        optimizer (Optional[partial[Optimizer]]): The optimizer to use during training.
        lr_scheduler (Optional[dict[str, partial[LRScheduler] | str]]): The learning rate scheduler to use during training.

    """

    def __init__(
        self,
        model: Module,
        encoder_decoder: BaseEncoderDecoder,
        pretraining_steps: int,
        inference_steps: int,
        scheduler: DSBScheduler,
        loss_fn: Optional[BaseLossFunction] = None,
        optimizer: Optional[partial[Optimizer]] = None,
        lr_scheduler: Optional[dict[str, partial[LRScheduler] | str]] = None,
        ema_decay: float = 0.999,
    ):
        super().__init__(optimizer, lr_scheduler)
        self.save_hyperparameters(
            ignore=["model", "encoder_decoder", "scheduler", "loss_fn", "optimizer", "lr_scheduler"]
        )
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.pretraining_steps = pretraining_steps
        self.inference_steps = inference_steps
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=ema_decay
        )  # keep a moving average of the model parameters for evaluation and sampling

    def to(self, device: torch.device):
        self.encoder_decoder.to(device)
        self.ema.to(device)
        return super().to(device)

    @property
    def pretraining(self) -> bool:
        return self.global_step < self.pretraining_steps

    @property
    def finetuning(self) -> bool:
        return not self.pretraining

    def forward(self, batch: SchedulerBatch) -> ModelOutput:
        output = self.model(batch["xt"], batch["timesteps"], batch["conditional"])
        return ModelOutput(output=output)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_decoder.encode(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.encoder_decoder.decode(z)

    def common_step(self, batch: UnpairedAudioBatch, batch_idx: int) -> StepOutput:
        """
        The training/validation/test step for the DSB model.
        The DSB model is provided with a UnpairedAudioBatch which consists of two unpaired batches, x0 and x1, from two different domains.

        During pretraining, the model tries to learn a path between these two random batches. This gives subpair results but is required for the model to converge.
        During finetuning, the model generates its own training examples by sampling from x0 to x1 and from x1 to x0 using the current model. This allows the model to learn from its own mistakes and improve over time.

        Args:
            batch (UnpairedAudioBatch): A batch of unpaired audio samples from two different domains
            batch_idx (int): The index of the batch in the current epoch.
        Returns:
            StepOutput: The output of the training step, containing the loss and any other relevant information

        """
        x0, x1 = batch["x0"], batch["x1"]
        x0, x1 = self.encode(x0), self.encode(x1)

        if self.pretraining:
            x0_b, x1_b = x0, x1
            x0_f, x1_f = x0, x1
        else:
            x0_b, x1_f = x0, x1
            x1_b = self.sample(x0_b, direction="forward", num_steps=self.inference_steps, verbose=False)
            x0_f = self.sample(x1_f, direction="backward", num_steps=self.inference_steps, verbose=False)

        scheduler_batch = self.scheduler.sample_training_batch(x0_b, x1_b, x0_f, x1_f)
        model_output = self(scheduler_batch)

        loss_output = self.loss_fn(model_output, scheduler_batch)

        return StepOutput(
            loss=loss_output["loss"],
            model_output=model_output,
            loss_output=loss_output,
            module=self,
        )

    def on_before_zero_grad(self, optimizer: Optimizer):
        # update ema weights
        self.ema.update()
        return super().on_before_zero_grad(optimizer)

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        # save ema state dict in checkpoint
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict):
        # load ema state dict from checkpoint
        self.ema.load_state_dict(checkpoint["ema"])

    @torch.no_grad()
    def sample(
        self,
        x_start: Tensor,
        direction: DIRECTIONS,
        num_steps: int,
        scheduler_type: SCHEDULER_TYPES = "linear",
        return_trajectory: bool = False,
        verbose: bool = False,
        encode: bool = False,
    ) -> Tensor:
        """
        Sampling from the model using the DSB sampling procedure.

        Args:
            x_start (Tensor): The starting point.
            direction (DIRECTIONS): The direction of sampling.
            num_steps (int): The number of steps to sample.
            scheduler_type (SCHEDULER_TYPES, optional): The type of scheduler to use. Defaults to "linear".
            return_trajectory (bool, optional): Whether to return the entire trajectory. Defaults to False.
            verbose (bool, optional): Whether to print progress. Defaults to False.
            encode (bool, optional): Whether to encode the input. Defaults to False.

        Returns:
            Tensor: The sampled output.
        """

        training = self.training  # Store the original training mode
        self.eval()  # Ensure the model is in eval mode for sampling

        if encode:
            x_start = self.encode(x_start)

        batch_size = x_start.shape[0]
        device = x_start.device
        c = self.scheduler.get_conditional(direction, batch_size, device)
        timeschedule = self.scheduler.get_timeschedule(num_steps, scheduler_type)
        # timeschedule is a list of tuples (tk_plus_one, tk) where tk_plus_one is the next timestep and tk is the current timestep, for example:
        # [(0.0, 0.5), (0.5, 1.0)] for a linear scheduler with 2 steps, where we first go from t=1.0 to t=0.5 and then from t=0.5 to t=0.0

        if direction == "backward":
            # normally, the timeschedule goes from t=0 to t=1, but for backward sampling we want to go from t=1 to t=0
            timeschedule = timeschedule[::-1]

        x = x_start.clone()
        trajectory = [self.decode(x) if encode else x]

        with self.ema.average_parameters():
            for tk_plus_one, tk in tqdm(timeschedule, desc="Sampling...", leave=False, disable=not verbose):
                t = torch.full((batch_size,), tk_plus_one if direction == "backward" else tk, device=device)
                x_in = torch.cat([x, x_start], dim=1) if self.scheduler.condition_on_start else x

                model_output = self(SchedulerBatch(xt=x_in, timesteps=t, conditional=c))

                x = self.scheduler.step(x, model_output["output"], tk_plus_one, tk, direction)
                trajectory.append(self.decode(x) if encode else x)

        if training:  # Restore the original training mode
            self.train()

        if return_trajectory:
            return torch.stack(trajectory, dim=0)

        return trajectory[-1]


@torch.compiler.disable
def wpe_preprocess(sample: Tensor):
    taps = 20
    delay = 3
    iterations = 5
    stft_size = 512
    stft_shift = 128

    device = sample.device
    processed = []

    for frame in sample:
        frame = frame.cpu().numpy()
        Y = stft(frame, stft_size, stft_shift)
        Y = Y.transpose(2, 0, 1)
        Z = wpe(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode="full")
        dereverb = istft(Z.transpose(1, 2, 0), size=stft_size, shift=stft_shift)
        processed.append(torch.tensor(dereverb, device=device))

    processed = torch.stack(processed).to(device=device, dtype=torch.float32)
    return processed


class DSBWithWPE(DSB):
    def __init__(
        self,
        model: Module,
        encoder_decoder: BaseEncoderDecoder,
        pretraining_steps: int,
        inference_steps: int,
        scheduler: DSBScheduler,
        loss_fn: Optional[BaseLossFunction] = None,
        optimizer: Optional[partial[Optimizer]] = None,
        lr_scheduler: Optional[dict[str, partial[LRScheduler] | str]] = None,
        ema_decay: float = 0.999,
    ):
        super().__init__(
            model=model,
            encoder_decoder=encoder_decoder,
            pretraining_steps=pretraining_steps,
            inference_steps=inference_steps,
            scheduler=scheduler,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_decay=ema_decay,
        )

    def common_step(self, batch: UnpairedAudioBatch, batch_idx: int) -> StepOutput:
        batch["x1"] = wpe_preprocess(batch["x1"])
        return super().common_step(batch, batch_idx)

    @torch.no_grad()
    def sample(
        self,
        x_start: Tensor,
        direction: DIRECTIONS,
        num_steps: int,
        scheduler_type: SCHEDULER_TYPES = "linear",
        return_trajectory: bool = False,
        verbose: bool = False,
        encode: bool = False,
    ) -> Tensor:
        if encode and direction == "backward":
            x_start = wpe_preprocess(x_start)

        return super().sample(
            x_start=x_start,
            direction=direction,
            num_steps=num_steps,
            scheduler_type=scheduler_type,
            return_trajectory=return_trajectory,
            verbose=verbose,
            encode=encode,
        )


class PairedDSB(DSB):
    def __init__(
        self,
        model: Module,
        encoder_decoder: BaseEncoderDecoder,
        scheduler: DSBScheduler,
        loss_fn: Optional[BaseLossFunction] = None,
        optimizer: Optional[partial[Optimizer]] = None,
        lr_scheduler: Optional[dict[str, partial[LRScheduler] | str]] = None,
    ):
        super().__init__(model, encoder_decoder, None, None, scheduler, loss_fn, optimizer, lr_scheduler)

    def common_step(self, batch: UnpairedAudioBatch, batch_idx: int) -> StepOutput:
        x0, x1 = batch["x0"], batch["x1"]
        x0, x1 = self.encode(x0), self.encode(x1)

        # in the paired setting, we only train the backward model and we use the original x0 and x1 as the training pairs
        scheduler_batch = self.scheduler._sample_training_batch(x0, x1, direction="backward")
        model_output = self(scheduler_batch)

        loss_output = self.loss_fn(model_output, scheduler_batch)

        return StepOutput(
            loss=loss_output["loss"],
            model_output=model_output,
            loss_output=loss_output,
            module=self,
        )

    def forward(self, batch: SchedulerBatch) -> ModelOutput:
        output = self.model(batch["xt"], batch["timesteps"])
        return ModelOutput(output=output)


if __name__ == "__main__":
    from src.networks.encoders import IdentityEncoderDecoder

    class DummyNetwork(Module):
        def __init__(self, output_tensor: Tensor):
            super().__init__()
            self.register_buffer("output_tensor", output_tensor)

        def forward(self, xt: Tensor, timesteps: Tensor, conditional: Tensor) -> Tensor:
            return self.output_tensor

    input_tensor = torch.randn(1, 10)
    output_tensor = torch.randn(1, 10)
    network = DummyNetwork(output_tensor)
    encoder_decoder = IdentityEncoderDecoder()
    scheduler = DSBScheduler(
        epsilon=1.0,
        target="terminal",
        condition_on_start=False,
    )

    dsb = DSB(
        model=network,
        encoder_decoder=encoder_decoder,
        pretraining_steps=10,  # doesn't matter for this test
        inference_steps=10,  # doesn't matter for this test
        scheduler=scheduler,
        loss_fn=None,  # doesn't matter for this test
        optimizer=None,  # doesn't matter for this test
        lr_scheduler=None,  # doesn't matter for this test
    )

    for direction in ["forward", "backward"]:
        sampled_tensor = dsb.sample(
            x_start=input_tensor,
            direction=direction,
            num_steps=10,
            scheduler_type="linear",
            return_trajectory=False,
            verbose=False,
            encode=False,
        )

        assert torch.allclose(
            sampled_tensor, output_tensor
        ), "The sampled tensor should be equal to the output tensor from the dummy network."
