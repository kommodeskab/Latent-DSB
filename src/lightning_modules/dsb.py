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
from .schedulers import DSBScheduler, BaseScheduler
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

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
        first_iteration : Literal['network', 'noise', 'pretrained'] = 'network',
        gamma_min : float = 1e-3,
        gamma_max : float = 1e-2,
        optimizer : Optimizer | None = None,
        target : Literal["terminal", "flow"] = "terminal",
        max_dsb_iterations : int | None = 10,
        max_norm : float = float("inf"),
        added_noise : float = 0.0,
        ema_decay : float = 0.999,
        pretrained_target : Literal["terminal", "flow"] | None = None,
        pretrained_gamma_frac : float | None = None,
    ):
        super().__init__()
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
        
        assert first_iteration in ["network", "noise", "pretrained"], "Invalid first_iteration"
        
        if first_iteration == "pretrained":
            assert pretrained_target is not None, "Pretrained target must be provided"
            assert pretrained_gamma_frac is not None, "Pretrained gamma fraction must be provided"
            self.first_iteration_scheduler = DSBScheduler(num_steps, pretrained_gamma_frac, pretrained_target)

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
        optims, schedulers = self.configure_optimizers()
        
        self.trainer.optimizers[0] = optims[0]
        self.trainer.optimizers[1] = optims[1]
        
        self.trainer.lr_scheduler_configs[0].scheduler = schedulers[0]['scheduler']
        self.trainer.lr_scheduler_configs[1].scheduler = schedulers[1]['scheduler']

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
        xk = x_start
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
            
        return self._sample(x_start, self.scheduler, forward, return_trajectory, show_progress, noise)
    
    def get_model(self, backward : bool) -> Tuple[Module, ExponentialMovingAverage]:
        model = self.backward_model if backward else self.forward_model
        ema = self.backward_ema if backward else self.forward_ema
        return model, ema

    def get_optimizer(self, backward : bool) -> tuple[Optimizer, LRScheduler]:
        backward_optim, forward_optim = self.optimizers()
        backward_scheduler, forward_scheduler = self.lr_schedulers()
        if backward:
            return backward_optim, backward_scheduler
        else:
            return forward_optim, forward_scheduler

    def _get_loss_name(self, backward : bool, is_training : bool):
        iteration = self.DSB_iteration
        direction = "backward" if backward else "forward"
        training = "train" if is_training else "val"
        return f"iteration_{iteration}/{direction}_loss/{training}"
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        training_backward = self.training_backward
        
        x0, x1 = batch
        terminal_points = x0 if training_backward else x1
                
        # adding a bit of noise to the latent augments new data but (usually) still decodes to the same image
        terminal_encoded = self.encode(terminal_points, add_noise = True)
        t1 = time.time()
        new_trajectory = self.sample(terminal_encoded, forward = training_backward, return_trajectory = True, noise='training')
        time_to_sample = time.time() - t1
        self.cache.add(new_trajectory.cpu())
        
        model, ema = self.get_model(backward = training_backward)
        model.train()
        optimizer, lr_scheduler = self.get_optimizer(backward = training_backward)
        
        for _ in range(self.hparams.cache_num_iters):
            trajectory = self.cache.sample().to(self.device)
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
            lr_scheduler.step()
            ema.update()
            
            self.curr_num_iters += 1

        model_name = "backward_model" if training_backward else "forward_model"
        
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
            batch = x0 if training_backward else x1
            batch = self.encode(batch)
            
            trajectory = self.sample(batch, forward = training_backward, return_trajectory = True, noise='training')
            
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
        backward_lr = backward_opt.param_groups[0]['lr']
        forward_lr = forward_opt.param_groups[0]['lr']
        
        backward_scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer = backward_opt, 
                T_max = self.hparams.max_iterations + self.hparams.cache_num_iters,
                eta_min = 0.1 * backward_lr
                ),
            'name': 'backward_lr'
        }
        forward_scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer = forward_opt, 
                T_max = self.hparams.max_iterations + self.hparams.cache_num_iters,
                eta_min = 0.1 * forward_lr
                ),
            'name': 'forward_lr'
        }
        
        return [backward_opt, forward_opt], [backward_scheduler, forward_scheduler]