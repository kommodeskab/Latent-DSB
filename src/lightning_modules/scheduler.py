from torch import Tensor
import torch
from typing import Literal, get_args
import math
from typing import Optional
from src import SchedulerBatch

DIRECTIONS = Literal["forward", "backward"]
SCHEDULER_TYPES = Literal["linear", "cosine"]
TARGETS = Literal["drift", "scaled_drift", "terminal"]


class DSBScheduler:
    def __init__(
        self,
        epsilon: float,
        target: TARGETS = "drift",
        condition_on_start: bool = False,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.target = target
        self.condition_on_start = condition_on_start

    def get_timeschedule(self, num_steps: int, scheduler_type: SCHEDULER_TYPES) -> list[tuple[float, float]]:
        """
        Time schedule for the DSB sampling procedure.
        The time schedule is a list of tuples, where each tuple contains the current time and the next time.

        Args:
            num_steps (int): The number of steps to sample.
            scheduler_type (SCHEDULER_TYPES): The type of scheduler to use. Can be either "linear" or "cosine".

        Raises:
            ValueError: If an unsupported scheduler type is provided. Use either: "linear" or "cosine".

        Returns:
            list[tuple[float, float]]: The time schedule for the DSB sampling procedure.
        """
        
        if scheduler_type == "linear":
            t = torch.linspace(0, 1, num_steps + 1)
        elif scheduler_type == "cosine":
            t = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, num_steps + 1)))
        else:
            raise ValueError(f"Unsupported time schedule type. Use either: {get_args(SCHEDULER_TYPES)}")

        t = t.tolist()
        timeschedule = [(t[i + 1], t[i]) for i in range(num_steps)]

        return timeschedule

    @staticmethod
    def to_dim(x: Tensor, dim: int) -> Tensor:
        """Utility function to ensure that the input tensor has the correct number of dimensions."""
        
        while x.dim() < dim:
            x = x.unsqueeze(-1)
        return x

    def get_conditional(self, direction: DIRECTIONS, batch_size: int, device: torch.device) -> Tensor:
        """Utility function to get the conditional tensor for the DSB sampling procedure. 
        This tells the model whether we are sampling in the forward or backward direction, and can be used by the model to condition its predictions accordingly."""
        if direction == "forward":
            return torch.ones((batch_size,), dtype=torch.long, device=device)
        elif direction == "backward":
            return torch.zeros((batch_size,), dtype=torch.long, device=device)
        else:
            raise ValueError(f"Unsupported direction. Use either: {get_args(DIRECTIONS)}")

    def _sample_training_batch(
        self, x0: Tensor, x1: Tensor, direction: DIRECTIONS, timesteps: Optional[Tensor] = None
    ) -> SchedulerBatch:
        """
        The sampling procedure for the DSB scheduler. 
        See this paper for example: https://arxiv.org/pdf/2409.09347
        
        Can be summarized like this:
        Given an initial and final point (x0, x1), we can sample a "intermediate" point at time t called x_t,
        as follows:
        x_t ~ p(x_t | x0, x1) = N(mu_t, sigma_t) where mu_t = (1 - t) * x0 + t * x1 and sigma_t = epsilon * t * (1 - t)
        Where epsilon is a hyperparameter that controls the amount of noise added to the system.
        
        The goal is then to learn a "backward" process, that somehow predicts the terminal point, i.e. where the process should end up
        (x0 if direction == "backward" and x1 if direction == "forward")
        
        """
        
        
        assert direction in get_args(DIRECTIONS), f"Unsupported direction. Use either: {get_args(DIRECTIONS)}"

        batch_size = x0.size(0)

        if timesteps is None:
            timesteps = torch.rand(batch_size, device=x0.device)

        timesteps = timesteps.clamp(max=0.999) if direction == "forward" else timesteps.clamp(min=0.001)

        t = self.to_dim(timesteps, x0.dim())
        mu_t = (1 - t) * x0 + t * x1
        sigma_t = self.epsilon * t * (1 - t)
        noise = torch.randn_like(mu_t)
        xt = mu_t + sigma_t**0.5 * noise

        accumulated_time = (1 - t) if direction == "forward" else t
        terminal = x1 if direction == "forward" else x0

        if self.target == "terminal":
            target = terminal
        elif self.target == "drift":
            target = (terminal - xt) / accumulated_time
        elif self.target == "scaled_drift":
            target = (terminal - xt) / accumulated_time.sqrt()

        conditional = self.get_conditional(direction, batch_size, x0.device)

        if self.condition_on_start:
            start = x0 if direction == "forward" else x1
            xt = torch.cat([xt, start], dim=1)

        return SchedulerBatch(
            xt=xt,
            target=target,
            timesteps=timesteps,
            conditional=conditional,
        )

    def sample_training_batch(self, x0_b: Tensor, x1_b: Tensor, x0_f: Tensor, x1_f: Tensor) -> SchedulerBatch:
        """
        Same as _sample_training_batch but here we both sample forward and backward process, so that the 
        model learns both objectives at the same time. This is good for gradients
        """
        
        b_batch = self._sample_training_batch(x0_b, x1_b, direction="backward")
        f_batch = self._sample_training_batch(x0_f, x1_f, direction="forward")

        xt = torch.cat([b_batch["xt"], f_batch["xt"]], dim=0)
        timesteps = torch.cat([b_batch["timesteps"], f_batch["timesteps"]], dim=0)
        conditional = torch.cat([b_batch["conditional"], f_batch["conditional"]], dim=0)
        target = torch.cat([b_batch["target"], f_batch["target"]], dim=0)

        return SchedulerBatch(
            xt=xt,
            target=target,
            timesteps=timesteps,
            conditional=conditional,
        )

    def step(self, xt: Tensor, model_output: Tensor, tk_plus_one: float, tk: float, direction: DIRECTIONS) -> Tensor:
        """
        The step function for the DSB sampling procedure.
        i.e. 
        p(x_{t-1} | x_t) if direction == "backward" and p(x_{t+1} | x_t) if direction == "forward"
        """
        
        
        assert direction in get_args(DIRECTIONS), f"Unsupported direction. Use either: {get_args(DIRECTIONS)}"
        assert tk_plus_one > tk, f"tk_plus_one ({tk_plus_one}) must be greater than tk ({tk})."

        delta_t = tk_plus_one - tk  # the step size
        accumulated_time = (1 - tk) if direction == "forward" else tk_plus_one

        if self.target == "terminal":
            drift = (model_output - xt) / accumulated_time
        elif self.target == "drift":
            drift = model_output
        elif self.target == "scaled_drift":
            drift = model_output / (accumulated_time**0.5)
        else:
            raise ValueError(f"Unsupported target type. Use either: {get_args(TARGETS)}")

        if direction == "backward":
            xnext_mean = xt + delta_t * drift
            xnext_var = self.epsilon * delta_t * tk / tk_plus_one
        elif direction == "forward":
            xnext_mean = xt + delta_t * drift
            xnext_var = self.epsilon * delta_t * (1 - tk_plus_one) / (1 - tk)

        noise = torch.randn_like(xnext_mean)

        xnext = xnext_mean + xnext_var**0.5 * noise

        return xnext
