import torch
from typing import Tuple, Literal
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from src.networks.encoders import BaseEncoderDecoder
from torch.optim import Optimizer
from torch.nn import Module
from pytorch_lightning.utilities import grad_norm
import copy
from src.lightning_modules.utils import Cache
from src.lightning_modules.fm import FM
from torch.nn.functional import mse_loss
from torch.nn import Module
import time
from .mixins import EncoderDecoderMixin
from .schedulers import DSBScheduler
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from src.utils import get_ckpt_path
from torch.optim.lr_scheduler import OneCycleLR

class DSB(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        num_steps : int,
        max_iterations : int,
        cache_max_size : int,
        cache_num_iters : int,
        use_pretrained_flow : bool = False,
        gamma_frac : float = 0.001,
        optimizer : Optimizer | None = None,
        backward_model : Module | None = None,
        target : Literal["terminal", "flow"] = "terminal",
        max_dsb_iterations : int | None = 10,
        max_norm : float = float("inf"),
        added_noise : float = 0.0,
        ema_decay : float = 0.999,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["model", "encoder_decoder", "backward_model"])
        
        self.training_backward = True
        self.curr_num_iters = 0
        self.DSB_iteration = 1
        self.added_noise = added_noise
        
        assert 0 < gamma_frac <= 1, "Gamma fraction must be in the range (0, 1]"
        self.scheduler = DSBScheduler(num_steps, gamma_frac, target)
        
        # if the backward model is not provided, make it a copy of the forward model 
        self.forward_model = model
        self.backward_model = backward_model if backward_model is not None else copy.deepcopy(model)
        
        # most often, we want to utilize a pretrained diffusion model for the first iterations
        if use_pretrained_flow:
            self.fmmodel = FM(model)
        
        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.cache = Cache(max_size = cache_max_size)
        
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

            if self.training_backward: 
                self._reset_optimizers()
                # save checkpoint under "logs/project/version/checkpoints"
                save_dir = f"logs/{self.logger.name}/{self.logger.version}/checkpoints/DSB_iteration_{self.DSB_iteration}.ckpt"
                self.trainer.save_checkpoint(save_dir)
                print(f"Saved checkpoint at {save_dir}")
                self.DSB_iteration += 1
                
            return -1
        
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
        backward_opt, forward_opt = optims
        backward_scheduler, forward_scheduler = schedulers
        self.trainer.optimizers[0] = backward_opt
        self.trainer.optimizers[1] = forward_opt
        self.trainer.lr_schedulers[0]['scheduler'] = backward_scheduler
        self.trainer.lr_schedulers[1]['scheduler'] = forward_scheduler

    @torch.no_grad()
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False, use_initial_forward_sampling : bool = False, show_progress : bool = False) -> Tensor:
        if use_initial_forward_sampling:
            if hasattr(self, "fmmodel"):
                return self.fmmodel.sample(x_start, return_trajectory)
            else:
                rand_noise = torch.randn_like(x_start)
                gammas_bar = self.scheduler.gammas_bar
                trajectory = [x_start]
                for k in range(self.hparams.num_steps):
                    x = gammas_bar[k + 1] * rand_noise + (1 - gammas_bar[k + 1]) * x_start
                    trajectory.append(x)
                    
                trajectory = torch.stack(trajectory, dim = 0)
                return trajectory if return_trajectory else x

        # otherwise, we sample using the diffusion schrödinger bridge
        xk = x_start
        trajectory = [xk]
        model, ema = self.get_model(backward = not forward)
        model.eval()
        for k in tqdm(reversed(self.scheduler.timesteps), desc='Sampling', disable=not show_progress):
            ks = torch.full((xk.size(0),), k, dtype=torch.int64, device=xk.device)
            with ema.average_parameters():
                model_output = model(xk, ks)
            xk = self.scheduler.step(xk, k, model_output)
            trajectory.append(xk)
            
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory if return_trajectory else trajectory[-1]
    
    def get_model(self, backward : bool) -> Tuple[Module, ExponentialMovingAverage]:
        model = self.backward_model if backward else self.forward_model
        ema = self.backward_ema if backward else self.forward_ema
        return model, ema
    
    def get_optimizer(self, backward : bool) -> tuple[Optimizer, OneCycleLR]:
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
        batch = x0 if training_backward else x1
        
        # if we are in the very first iteration, we use the initial sampling
        use_initial_forward_sampling = training_backward and self.DSB_iteration == 1
        
        # adding a bit of noise to the latent augments new data but (usually) still decodes to the same image
        batch = self.encode(batch, add_noise = True)
        t1 = time.time()
        trajectory = self.sample(batch, forward = training_backward, return_trajectory = True, use_initial_forward_sampling = use_initial_forward_sampling)
        time_to_sample = time.time() - t1
        self.cache.add(trajectory)
        
        model, _ = self.get_model(backward = training_backward)
        model.train()
        optimizer, lr_scheduler = self.get_optimizer(backward = training_backward)
                
        for _ in range(self.hparams.cache_num_iters):
            trajectory = self.cache.sample()
            xt, timesteps, target = self.scheduler.sample_batch(trajectory)
            model_output = model(xt, timesteps)

            # calculate loss and do backward pass
            optimizer.zero_grad()
            # if training backward, then sampled_batch = xk + 1 else sampled_batch = xk
            loss = mse_loss(model_output, target)
            self.manual_backward(loss)
            
            # clip the gradients. first, save the norm for later logging
            norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.clip_gradients(optimizer, self.hparams.max_norm, "norm")
            optimizer.step()
            lr_scheduler.step()
            
            self.curr_num_iters += 1

        model_name = "backward_model" if training_backward else "forward_model"
        
        self.log_dict({
            self._get_loss_name(backward = training_backward, is_training = True): loss,
            f"{model_name}_grad": norm,
            "curr_num_iters": self.curr_num_iters,
            "DSB_iteration": self.DSB_iteration,
            "time_to_sample": time_to_sample,
            "training_backward": training_backward,
        }, on_step=True)

    @torch.no_grad()
    def validation_step(self, batch : Tensor, batch_idx : int) -> None:
        torch.manual_seed(0)
        is_backward = self.training_backward
        x0, x1 = batch
        batch = x0 if is_backward else x1
        batch = self.encode(batch)
        use_initial_forward_sampling = is_backward and self.DSB_iteration == 1
        model, ema = self.get_model(is_backward)
        model.eval()
        
        trajectory = self.sample(batch, forward = is_backward, return_trajectory = True, use_initial_forward_sampling=use_initial_forward_sampling)
        for _ in range(self.hparams.cache_num_iters):
            xk, timesteps, target = self.scheduler.sample_batch(trajectory)
            with ema.average_parameters():
                model_output = model(xk, timesteps)
            loss = mse_loss(model_output, target)
            self.log_dict({
                self._get_loss_name(backward = is_backward, is_training = False): loss,
            }, on_step=False, on_epoch=True)
    
    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        # make the optimizers
        backward_opt = self.partial_optimizer(self.backward_model.parameters())
        forward_opt = self.partial_optimizer(self.forward_model.parameters())
        lr = backward_opt.param_groups[0]['lr']
        
        backward_scheduler = {
            'scheduler': OneCycleLR(
                backward_opt, max_lr=10*lr, 
                total_steps=self.hparams.max_iterations
                ),
            'name': 'backward_lr'
        }
        forward_scheduler = {
            'scheduler': OneCycleLR(
                forward_opt, max_lr=10*lr, 
                total_steps=self.hparams.max_iterations
                ),
            'name': 'forward_lr'
        }
        
        return [backward_opt, forward_opt], [backward_scheduler, forward_scheduler]