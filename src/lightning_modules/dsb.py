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
from .schedulers import DSBScheduler, NOISE_TYPES, TARGETS
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset

class DSB(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        backward_model : Module,
        forward_model : Module,
        encoder_decoder : BaseEncoderDecoder,
        effective_batch_size : int,
        num_timesteps : int = 100,
        cache_max_size : int = 5_000,
        max_iterations : int = 32_000,
        cache_num_iters : int = 20,
        first_iteration : Literal['network', 'network-straight', 'noise', 'pretrained'] = 'network',
        gamma_min : float | None = None,
        gamma_max : float | None = None,
        optimizer : Optimizer | None = None,
        target : TARGETS = "terminal",
        max_norm : float = 5.0,
        ema_decay : float = 0.999,
        pretrained_target : TARGETS | None = None,
    ):
        super().__init__()
        assert first_iteration in ["network", "network-straight", "noise", "pretrained"], "Invalid first_iteration"
        
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["backward_model", "forward_model", "encoder_decoder"])
        
        self.training_backward = True
        self.curr_num_iters = 0
        self.DSB_iteration = 1
        self.first_iteration = first_iteration
        
        self.scheduler = DSBScheduler(num_timesteps, gamma_min, gamma_max, target)
        
        # if the backward model is not provided, make it a copy of the forward model 
        self.forward_model = forward_model
        self.backward_model = backward_model
        
        if first_iteration == "pretrained":
            assert pretrained_target is not None, "Pretrained target must be provided"
            self.first_iteration_scheduler = DSBScheduler(num_timesteps, gamma_min, gamma_max, pretrained_target)

        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.cache = DSBCache(max_size=cache_max_size, batch_size=effective_batch_size)
        
        self.forward_ema = ExponentialMovingAverage(self.forward_model.parameters(), decay=ema_decay)
        self.backward_ema = ExponentialMovingAverage(self.backward_model.parameters(), decay=ema_decay)
        
    def on_fit_start(self) -> None:
        self.forward_ema.to(self.device)
        self.backward_ema.to(self.device)
        self.encoder_decoder.to(self.device)
        
    def on_train_batch_start(self, batch : Tensor, batch_idx : int) -> None:
        if self.curr_num_iters >= self.hparams.max_iterations:
            print(f"Max iteration reached for DSB iteration {self.DSB_iteration} / training backward: {self.training_backward}.")
            old_optim = self.get_optimizer(self.training_backward)
            self.untoggle_optimizer(old_optim)
            
            self.curr_num_iters = 0
            self.training_backward = not self.training_backward
            self.cache.clear()
            
            new_optim = self.get_optimizer(self.training_backward)
            self.toggle_optimizer(new_optim)
            
            self.forward_ema.copy_to()
            self.backward_ema.copy_to()

            if self.training_backward: 
                self._reset_optimizers()
                self.DSB_iteration += 1
                
            print(f"Starting DSB iteration {self.DSB_iteration} / training backward: {self.training_backward}.")
                
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
        noise : NOISE_TYPES = 'inference',
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
        noise : NOISE_TYPES = 'inference',
    ) -> Tensor:
        if (self.training_backward and self.DSB_iteration == 1 and forward): # when sampling forward in the first iteration
            if self.first_iteration == 'noise':
                x_end = torch.randn_like(x_start)
                return self.scheduler.deterministic_sample(x_start, x_end, return_trajectory, noise=noise)
            
            elif self.first_iteration == 'pretrained':
                return self._sample(x_start, self.first_iteration_scheduler, forward, return_trajectory, show_progress, noise)
        
        if self.first_iteration == 'network-straight':
            x_end = self._sample(x_start, self.scheduler, forward, False, show_progress, 'inference')
            return self.scheduler.deterministic_sample(x_start, x_end, return_trajectory, noise=noise)
        
        return self._sample(x_start, self.scheduler, forward, return_trajectory, show_progress, noise)
    
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
            
        terminal_encoded = self.encode(terminal_points)
        new_trajectory = self.sample(terminal_encoded, forward = training_backward, return_trajectory = True, noise='training', show_progress=False)
        self.cache.add(new_trajectory.cpu())
  
        model, ema = self.get_model(backward = training_backward)
        model.train()
        optimizer = self.get_optimizer(backward = training_backward)
        
        num_iters = self.hparams.cache_num_iters if self.cache.is_full() else 1
        
        for i in range(num_iters):
            if i == 0:
                trajectory = new_trajectory[:, :8]
            else:
                trajectory = self.cache.sample().to(device=self.device)               
            
            xt, timesteps, target = self.scheduler.sample_batch(trajectory)
            
            model_output = model(xt, timesteps)
            loss = mse_loss(model_output, target)
            
            optimizer.zero_grad()
            self.manual_backward(loss)
            
            # clip the gradients. first, save the norm for later logging
            norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.clip_gradients(optimizer, self.hparams.max_norm, "norm")
            optimizer.step()
            ema.update()
            
            self.curr_num_iters += 1
            
            if i == 0:
                model_name = "backward_model" if training_backward else "forward_model"
                self.log_dict({
                    self._get_loss_name(backward = training_backward, is_training = True): loss,
                    f"{model_name}_grad": norm,
                    "curr_num_iters": self.curr_num_iters,
                    "DSB_iteration": self.DSB_iteration,
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
            batch_size = trajectory.size(1)
            
            model, ema = self.get_model(training_backward)
            model.eval()
            
            for k in self.scheduler.timesteps:
                timesteps = torch.full((batch_size,), k, dtype=torch.int64, device=trajectory.device)
                xk, timesteps, target = self.scheduler.sample_batch(trajectory, timesteps)
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