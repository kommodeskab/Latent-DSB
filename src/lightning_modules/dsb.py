from torch import Tensor
import torch
from src import UnpairedBatch, ModelOutput, StepOutput, SchedulerBatch
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
        self.save_hyperparameters(ignore=["model", "encoder_decoder", "loss_fn", "optimizer", "lr_scheduler"])
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.pretraining_steps = pretraining_steps
        self.inference_steps = inference_steps
        self.loss_fn = loss_fn
        self.scheduler = scheduler

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

    def common_step(self, batch: UnpairedBatch, batch_idx: int) -> StepOutput:
        x0, x1 = batch["x0"], batch["x1"]
        x0, x1 = self.encode(x0), self.encode(x1)

        if self.pretraining:
            x0_b, x1_b = x0, x1
            x0_f, x1_f = x0, x1
        else:
            x0_b, x1_f = x0, x1
            x1_b = self.sample(x0_b, direction="forward", num_steps=self.inference_steps)
            x0_f = self.sample(x1_f, direction="backward", num_steps=self.inference_steps)

        scheduler_batch = self.scheduler.sample_training_batch(x0_b, x1_b, x0_f, x1_f)
        model_output = self.forward(scheduler_batch)

        loss_output = self.loss_fn.forward(model_output, scheduler_batch)

        return StepOutput(
            loss=loss_output["loss"],
            model_output=model_output,
            loss_output=loss_output,
            module=self,
        )

    @torch.no_grad()
    def sample(
        self,
        x_start: Tensor,
        direction: DIRECTIONS,
        num_steps: int,
        scheduler_type: SCHEDULER_TYPES = "linear",
        return_trajectory: bool = False,
        verbose: bool = True,
    ) -> Tensor:
        self.model.eval()

        batch_size = x_start.shape[0]
        device = x_start.device
        c = self.scheduler.get_conditional(direction, batch_size, device)
        timeschedule = self.scheduler.get_timeschedule(num_steps, scheduler_type)

        if direction == "forward":
            timeschedule = timeschedule[::-1]

        x = x_start.clone()
        trajectory = [x]

        for tk_plus_one, tk in tqdm(timeschedule, desc="Sampling...", leave=False, disable=not verbose):
            t = torch.full((batch_size,), tk_plus_one if direction == "backward" else tk, device=device)
            x_in = torch.cat([x, x_start], dim=1) if self.scheduler.condition_on_start else x
            model_output = self.forward(SchedulerBatch(xt=x_in, timesteps=t, conditional=c))
            x = self.scheduler.step(x, model_output["output"], tk_plus_one, tk, direction)
            trajectory.append(x)

        self.model.train()

        if return_trajectory:
            return torch.stack(trajectory, dim=0)

        return x
