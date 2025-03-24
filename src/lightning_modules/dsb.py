import torch
from typing import Tuple, Literal
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from src.networks.encoders import BaseEncoderDecoder
from torch.optim import Optimizer
from pytorch_lightning.utilities import grad_norm
from src.lightning_modules.utils import DSBCache
from torch.nn.functional import mse_loss
from torch.nn import Module
import time
from .mixins import EncoderDecoderMixin
from .schedulers import DSBScheduler
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import wandb
from matplotlib.figure import Figure

def plot_images(samples : Tensor, height : int | None = None, width : int | None = None) -> Figure:
    # assume samples have shape (k, c, h, w)
    # and have values between -1 and 1
    if height is None and width is None:
        k = int(samples.size(0) ** 0.5)
        height = width = k

    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    cmap = 'gray' if samples.shape[1] == 1 else None
    samples = samples.permute(0, 2, 3, 1)
    fig, axs = plt.subplots(height, width, figsize=(width*5, height*5), dpi=300)
    axs : list[plt.Axes]
    for i in range(height):
        for j in range(width):
            ax = axs[i, j] if height > 1 else axs[j]
            ax.imshow(samples[i * height + j], cmap=cmap)
            ax.axis('off')
    return fig

class DSB(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        backward_model : Module,
        forward_model : Module,
        encoder_decoder : BaseEncoderDecoder,
        num_steps : int = 100,
        cache_max_size : int = 5000,
        max_iterations : int = 20000,
        cache_num_iters : int = 20,
        first_iteration : Literal['network', 'network-straight', 'noise', 'pretrained'] = 'network',
        gamma_min : float | None = None,
        gamma_max : float | None = None,
        optimizer : Optimizer | None = None,
        target : Literal["terminal", "flow", "semi-flow"] = "terminal",
        max_dsb_iterations : int | None = 10,
        max_norm : float = float("inf"),
        added_noise : float = 0.0,
        ema_decay : float = 0.999,
        pretrained_target : Literal["terminal", "flow", "semi-flow"] | None = None,
    ):
        super().__init__()
        assert first_iteration in ["network", "network-straight", "noise", "pretrained"], "Invalid first_iteration"
        
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["backward_model", "forward_model", "encoder_decoder"])
        
        self.training_backward = True
        self.curr_num_iters = 0
        self.DSB_iteration = 1
        self.added_noise = added_noise
        self.first_iteration = first_iteration
        
        self.scheduler = DSBScheduler(num_steps, gamma_min, gamma_max, target)
        
        # if the backward model is not provided, make it a copy of the forward model 
        self.forward_model = forward_model
        self.backward_model = backward_model
        
        if first_iteration == "pretrained":
            assert pretrained_target is not None, "Pretrained target must be provided"
            self.first_iteration_scheduler = DSBScheduler(num_steps, gamma_min, gamma_max, pretrained_target)

        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.cache = DSBCache(max_size=cache_max_size)
        
        self.forward_ema = ExponentialMovingAverage(self.forward_model.parameters(), decay=ema_decay)
        self.backward_ema = ExponentialMovingAverage(self.backward_model.parameters(), decay=ema_decay)
        
    def on_fit_start(self) -> None:
        self.forward_ema.to(self.device)
        self.backward_ema.to(self.device)
        
    def _has_converged(self) -> bool:
        dsb_iters = self.DSB_iteration
        curr_iters = self.curr_num_iters
        max_iters = self.hparams.max_iterations
        max_dsb_iters = self.hparams.max_dsb_iterations
        return (curr_iters >= max_iters) or (dsb_iters >= max_dsb_iters)
    
    def on_train_batch_start(self, batch : Tensor, batch_idx : int) -> None:
        if self._has_converged():
            self.curr_num_iters = 0
            self.training_backward = not self.training_backward
            self.cache.clear()
            
            self.forward_ema.copy_to()
            self.backward_ema.copy_to()

            if self.training_backward: 
                self._reset_optimizers()
                self.DSB_iteration += 1
                
            return -1
        
    def on_validation_end(self):
        # save checkpoint under "logs/project/version/checkpoints"
        save_dir = f"logs/{self.logger.name}/{self.logger.version}/checkpoints/DSB_iteration_{self.DSB_iteration}.ckpt"
        self.trainer.save_checkpoint(save_dir)
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['DSB_iteration'] = self.DSB_iteration
        checkpoint['curr_num_iters'] = self.curr_num_iters
        checkpoint['training_backward'] = self.training_backward
        checkpoint['forward_ema'] = self.forward_ema.state_dict()
        checkpoint['backward_ema'] = self.backward_ema.state_dict()
        
    def on_load_checkpoint(self, checkpoint):
        self.DSB_iteration = checkpoint['DSB_iteration']
        self.curr_num_iters = checkpoint['curr_num_iters']
        self.training_backward = checkpoint['training_backward']
        self.forward_ema.load_state_dict(checkpoint['forward_ema'])
        self.backward_ema.load_state_dict(checkpoint['backward_ema'])
        
    def _reset_optimizers(self) -> None:
        backward_optim, forward_optim = self.configure_optimizers()
        
        self.trainer.optimizers[0] = backward_optim
        self.trainer.optimizers[1] = forward_optim

    @torch.no_grad()
    def _sample(
        self, 
        x_start : Tensor, 
        scheduler : DSBScheduler, 
        forward : bool = True, 
        return_trajectory : bool = False, 
        show_progress : bool = False,
        noise : Literal['training', 'inference', 'none'] = 'inference',
    ) -> Tensor:
        model, ema = self.get_model(backward = not forward)
        model.eval()
        xk = x_start.clone()
        trajectory = [xk]
        batch_size = xk.size(0)
        for k in tqdm(reversed(scheduler.timesteps), desc='Sampling', disable=not show_progress):
            ks = torch.full((batch_size,), k, dtype=torch.int64, device=xk.device)
            with ema.average_parameters():
                model_output = model(xk, ks)
            xk = scheduler.step(xk, k, model_output, noise)
            trajectory.append(xk)
            
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory if return_trajectory else trajectory[-1]

    @torch.no_grad()
    def sample(
        self, 
        x_start : Tensor, 
        forward : bool = True, 
        return_trajectory : bool = False, 
        show_progress : bool = False,
        noise : Literal['training', 'inference', 'none'] = 'inference',
    ) -> Tensor:
        if (self.training_backward and self.DSB_iteration == 1 and forward): # when sampling forward in the first iteration
            if self.first_iteration == 'noise':
                x_end = torch.randn_like(x_start).to(x_start.device)
                return self.scheduler.deterministic_sample(x_start, x_end, return_trajectory, noise=noise)
            
            elif self.first_iteration == 'pretrained':
                return self._sample(x_start, self.first_iteration_scheduler, forward, return_trajectory, show_progress, noise)
        
        if self.first_iteration == 'network':
            return self._sample(x_start, self.scheduler, forward, return_trajectory, show_progress, noise)
        
        if self.first_iteration == 'network-straight':
            x_end = self._sample(x_start, self.scheduler, forward, False, show_progress, 'inference')
            return self.scheduler.deterministic_sample(x_start, x_end, return_trajectory, noise=noise)
    
    def get_model(self, backward : bool) -> Tuple[Module, ExponentialMovingAverage]:
        model = self.backward_model if backward else self.forward_model
        ema = self.backward_ema if backward else self.forward_ema
        return model, ema

    def get_optimizer(self, backward : bool) -> Optimizer:
        backward_optim, forward_optim = self.optimizers()
        return backward_optim if backward else forward_optim

    def _get_loss_name(self, backward : bool, is_training : bool):
        iteration = self.DSB_iteration
        direction = "backward" if backward else "forward"
        training = "train" if is_training else "val"
        return f"iteration_{iteration}/{direction}_loss/{training}"
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        training_backward = self.training_backward
        
        x0, x1 = batch
        terminal_points = x0 if training_backward else x1
                
        terminal_encoded = self.encode(terminal_points, add_noise = True)
        t1 = time.time()
        new_trajectory = self.sample(terminal_encoded, forward = training_backward, return_trajectory = True, noise='training')
        time_to_sample = time.time() - t1
        self.cache.add(new_trajectory.cpu())
        
        if self.curr_num_iters == 0:
            logger = self.logger
            idxs_to_log = torch.linspace(0, new_trajectory.size(0) - 1, 5).long()
            trajectory_to_log = new_trajectory[idxs_to_log, :5].flatten(0, 1)
            trajectory_to_log = self.decode(trajectory_to_log)
            trajectory_to_log = trajectory_to_log.cpu()

            if x0.dim() == 4:
                fig = plot_images(trajectory_to_log, 5, 5,)
                key = f"iteration_{self.DSB_iteration}"
                key = f"{key}/forward_sample_for_backward_training" if training_backward else f"{key}/backward_sample_for_forward_training"
                logger.log_image(
                    key=key,
                    images=[wandb.Image(fig)],
                )
                plt.close(fig)
        
        model, ema = self.get_model(backward = training_backward)
        model.train()
        optimizer = self.get_optimizer(backward = training_backward)
        
        for i in range(self.hparams.cache_num_iters):
            trajectory = new_trajectory if i == 0 else self.cache.sample().to(self.device)                
            xt, timesteps, target = self.scheduler.sample_batch(trajectory)
            model_output = model(xt, timesteps)

            # calculate loss and do backward pass
            loss = mse_loss(model_output, target)
            optimizer.zero_grad()
            self.manual_backward(loss)
            
            # clip the gradients. first, save the norm for later logging
            norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.clip_gradients(optimizer, self.hparams.max_norm, "norm")
            optimizer.step()
            ema.update()
            
            self.curr_num_iters += 1
            
            model_name = "backward_model" if training_backward else "forward_model"
            if i == 0:
                self.log_dict({
                    self._get_loss_name(backward = training_backward, is_training = True): loss,
                    f"{model_name}_grad": norm,
                    "curr_num_iters": self.curr_num_iters,
                    "DSB_iteration": self.DSB_iteration,
                    "time_to_sample": time_to_sample,
                    "training_backward": training_backward,
                    "cache_size": len(self.cache),
                }, on_step=True, prog_bar=True)


    @torch.no_grad()
    def validation_step(self, batch : Tensor, batch_idx : int) -> None:
        with self.fix_validation_seed():
            training_backward = self.training_backward
            x0, x1 = batch
            terminal_point = x0 if training_backward else x1
            terminal_encoded = self.encode(terminal_point)
            
            trajectory = self.sample(terminal_encoded, forward = training_backward, return_trajectory = True, noise='training')
            
            model, ema = self.get_model(training_backward)
            model.eval()
            
            for _ in range(self.hparams.cache_num_iters):
                xk, timesteps, target = self.scheduler.sample_batch(trajectory)
                with ema.average_parameters():
                    model_output = model(xk, timesteps)
                loss = mse_loss(model_output, target)
                self.log_dict({
                    "curr_num_iters": self.curr_num_iters,
                    self._get_loss_name(backward = training_backward, is_training = False): loss,
                }, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self) -> Tuple[list[Optimizer], list[dict]]:
        # make the optimizers
        backward_opt : Optimizer = self.partial_optimizer(self.backward_model.parameters())
        forward_opt : Optimizer = self.partial_optimizer(self.forward_model.parameters())
        return backward_opt, forward_opt