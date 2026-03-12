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
from .scheduler import DSBScheduler, DIRECTIONS, SCHEDULER_TYPES
from torch_ema import ExponentialMovingAverage
from contextlib import nullcontext


class DSB(BaseLightningModule):
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
    ):
        super().__init__(optimizer, lr_scheduler)
        self.save_hyperparameters(ignore=["model", "encoder_decoder", "scheduler", "loss_fn", "optimizer", "lr_scheduler"])
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.pretraining_steps = pretraining_steps
        self.inference_steps = inference_steps
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.9999) # keep a moving average of the model parameters for evaluation and sampling

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
        context = self.ema.average_parameters() if not self.training else nullcontext()
        
        with context:
            output = self.model(batch["xt"], batch["timesteps"], batch["conditional"])
            
        return ModelOutput(output=output)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_decoder.encode(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.encoder_decoder.decode(z)

    def common_step(self, batch: UnpairedAudioBatch, batch_idx: int) -> StepOutput:
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
        model_output = self.forward(scheduler_batch)

        loss_output = self.loss_fn.forward(model_output, scheduler_batch)

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

        for tk_plus_one, tk in tqdm(timeschedule, desc="Sampling...", leave=False, disable=not verbose):
            t = torch.full((batch_size,), tk_plus_one if direction == "backward" else tk, device=device)
            x_in = torch.cat([x, x_start], dim=1) if self.scheduler.condition_on_start else x
            
            with torch.no_grad():
                model_output = self.forward(SchedulerBatch(xt=x_in, timesteps=t, conditional=c))
                
            x = self.scheduler.step(x, model_output["output"], tk_plus_one, tk, direction)
            trajectory.append(self.decode(x) if encode else x)

        if training:  # Restore the original training mode
            self.train()
            
        if return_trajectory:
            return torch.stack(trajectory, dim=0)

        return trajectory[-1]
