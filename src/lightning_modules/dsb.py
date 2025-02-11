import torch
from typing import Tuple, Literal
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from src.networks.encoders import BaseEncoderDecoder
from torch.optim import Optimizer
from torch.nn import Module
from pytorch_lightning.utilities import grad_norm
import copy
from src.lightning_modules.utils import Cache, GammaScheduler
from src.lightning_modules.fm import FM
from torch.nn.functional import mse_loss
from torch.nn import Module
from src.data_modules.base_dm import BaseDSBDM
import time

class DSB(BaseLightningModule):
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        num_steps : int,
        max_iterations : int,
        cache_max_size : int,
        cache_num_iters : int,
        use_pretrained_flow : bool = False,
        gamma_frac : float = 1.0,
        optimizer : Optimizer | None = None,
        backward_model : Module | None = None,
        target : Literal["terminal", "flow"] = "terminal",
        max_dsb_iterations : int | None = 10,
        max_norm : float = float("inf"),
        lr_multiplier : float = 1.0,
        added_noise : float = 0.1,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["model", "encoder_decoder", "backward_model"])
        
        self.training_backward = True
        self.curr_num_iters = 0
        self.DSB_iteration = 1
        self.added_noise = added_noise
        
        assert 0 < gamma_frac <= 1, "Gamma fraction must be in the range (0, 1]"
        self.gamma_scheduler = GammaScheduler(gamma_frac, 1, num_steps, 1)
        
        # if the backward model is not provided, make it a copy of the forward model 
        self.forward_model = model
        self.backward_model = backward_model if backward_model is not None else copy.deepcopy(model)
        
        # most often, we want to utilize a pretrained diffusion model for the first iterations
        if use_pretrained_flow:
            self.fmmodel = FM(model)
            self.fmmodel.scheduler.set_timesteps(num_steps, gamma_frac)
        
        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.cache = Cache(max_size = cache_max_size)
        
    @property
    def datamodule(self) -> BaseDSBDM:
        return self.trainer.datamodule

    def on_fit_start(self) -> None:
        self.datamodule.training_backward = self.training_backward
        assert self.trainer.reload_dataloaders_every_n_epochs == 1, "The trainer must reload dataloaders every epoch for the DSB algorithm to work."

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
            self.datamodule.training_backward = self.training_backward
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
        
    def on_load_checkpoint(self, checkpoint):
        self.DSB_iteration = checkpoint['DSB_iteration']
        self.curr_num_iters = checkpoint['curr_num_iters']
        self.training_backward = checkpoint['training_backward']
        
    def _reset_optimizers(self) -> None:
        backward_opt, forward_opt = self.configure_optimizers()
        lr = backward_opt.param_groups[0]['lr']
        new_lr = lr * self.hparams.lr_multiplier ** (self.DSB_iteration - 1)
        for param_group in backward_opt.param_groups:
            param_group['lr'] = new_lr
        for param_group in forward_opt.param_groups:
            param_group['lr'] = new_lr
        self.trainer.optimizers[0] = backward_opt
        self.trainer.optimizers[1] = forward_opt
    
    def k_to_tensor(self, k : int, size : Tuple[int]) -> Tensor:
        return torch.full((size, ), k, dtype = torch.int64, device = self.device)
    
    def forward_call(self, x : Tensor, k: Tensor) -> Tensor:
        return self.forward_model(x, k)
    
    def backward_call(self, x : Tensor, k : Tensor) -> Tensor:
        return self.backward_model(x, k)
    
    @torch.no_grad()
    def encode(self, x : Tensor, add_noise : bool = False) -> Tensor:
        encoded = self.encoder_decoder.encode(x)
        if add_noise:
            encoded += self.added_noise * torch.randn_like(encoded)
        return encoded
    
    @torch.no_grad()
    def decode(self, x : Tensor) -> Tensor:
        return self.encoder_decoder.decode(x)
    
    @torch.no_grad()
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        """        
        Get :math:`x_{k + 1} | x_{k}`
        
        :param Tensor xk: the current point
        :param int k: the current step
        
        :return xk_plus_one: the next point
        """
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        pred = self.forward_call(xk, ks)
        sch = self.gamma_scheduler
        
        if self.hparams.target == "terminal":
            mu = xk + sch.gammas[k + 1] / (sch.gammas_bar[-1] - sch.gammas_bar[k]) * (pred - xk)
            sigma = torch.sqrt(sch.sigma_forward[k + 1])
        
        xk_plus_one = mu + torch.sqrt(sigma) * torch.randn_like(xk)
        return xk_plus_one
    
    @torch.no_grad()
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        """
        Get :math:`x_{k} | x_{k + 1}`
        
        :param Tensor xk_plus_one: the current point
        :param int k_plus_one: the current step
        
        :return xk: the previous point
        """
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        pred = self.backward_call(xk_plus_one, ks_plus_one)
        sch = self.gamma_scheduler
        
        if self.hparams.target == "terminal":
            mu = xk_plus_one + sch.gammas[k_plus_one] / sch.gammas_bar[k_plus_one] * (pred - xk_plus_one)
            sigma = torch.sqrt(sch.sigma_backward[k_plus_one])
            
        xk = mu + torch.sqrt(sigma) * torch.randn_like(xk_plus_one)
        return xk
    
    def _forward_loss(self, xk : Tensor, ks : int, xN : Tensor) -> Tensor:
        pred = self.forward_call(xk, ks)
        if self.hparams.target == "terminal":
            loss = mse_loss(pred, xN)
        
        return loss
    
    def _backward_loss(self, xk_plus_one : Tensor, ks_plus_one : int, x0 : Tensor) -> Tensor:
        pred = self.backward_call(xk_plus_one, ks_plus_one)
        if self.hparams.target == "terminal":
            loss = mse_loss(pred, x0)
            
        return loss
    
    @torch.no_grad()
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False, use_initial_forward_sampling : bool = False) -> Tensor:
        if use_initial_forward_sampling:
            if hasattr(self, "fmmodel"):
                return self.fmmodel.sample(x_start, return_trajectory)
            else:
                rand_noise = torch.randn_like(x_start)
                gammas_bar = self.gamma_scheduler.gammas_bar
                trajectory = [x_start]
                for k in range(self.hparams.num_steps):
                    t = gammas_bar[k + 1] / gammas_bar[-1]
                    x = t * rand_noise + (1 - t) * x_start
                    trajectory.append(x)
                    
                trajectory = torch.stack(trajectory, dim = 0)
                return trajectory if return_trajectory else x

        # otherwise, we sample using the diffusion schrödinger bridge
        xk = x_start
        trajectory = [xk]
        model = self._get_model(forward)
        model.eval()
        range_ = range(self.hparams.num_steps)
        if not forward:
            range_ = reversed(range_)
        
        for k in range_:
            xk_next = self.go_forward(xk, k) if forward else self.go_backward(xk, k + 1)
            trajectory.append(xk_next)
            xk = xk_next
        
        trajectory = trajectory[::-1] if not forward else trajectory
        trajectory = torch.stack(trajectory, dim = 0)

        return trajectory if return_trajectory else xk
    
    def _get_model(self, backward : bool) -> Module:
        return self.backward_model if backward else self.forward_model
    
    def _get_optimizer(self, backward : bool) -> Optimizer:
        backward_opt, forward_opt = self.optimizers()
        return backward_opt if backward else forward_opt

    def _get_loss_name(self, backward : bool, is_training : bool):
        iteration = self.DSB_iteration
        direction = "backward" if backward else "forward"
        training = "train" if is_training else "val"
        return f"iteration_{iteration}/{direction}_loss/{training}"
    
    def get_batch(self, backward : bool, trajectory : Tensor | None = None) -> Tensor:
        if trajectory is None:
            trajectory = self.cache.sample()
        batch_size = trajectory.size(1)
        random_samples = torch.randint(0, batch_size, (batch_size,)).type_as(trajectory)
        random_ks = torch.randint(0, self.hparams.num_steps, (batch_size,)).type_as(trajectory)
        if backward:
            random_ks += 1
        sampled_batch = trajectory[random_ks, random_samples]
        x0, xN = trajectory[0][random_samples], trajectory[-1][random_samples]
        return sampled_batch, random_ks, x0, xN
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        training_backward = self.training_backward
        
        # if we are in the very first iteration, we use the initial sampling
        use_initial_forward_sampling = training_backward and self.DSB_iteration == 1
        
        # adding a bit of noise to the latent augments new data but (usually) still decodes to the same image
        batch = self.encode(batch, add_noise = True)
        t1 = time.time()
        trajectory = self.sample(batch, forward = training_backward, return_trajectory = True, use_initial_forward_sampling = use_initial_forward_sampling)
        time_to_sample = time.time() - t1
        self.cache.add(trajectory)
        
        model = self._get_model(training_backward)
        optimizer = self._get_optimizer(training_backward)
                
        for i in range(self.hparams.cache_num_iters):
            sampled_batch, ks, x0, xN = self.get_batch(training_backward, trajectory if i == 0 else None)

            # calculate loss and do backward pass
            optimizer.zero_grad()
            # if training backward, then sampled_batch = xk + 1 else sampled_batch = xk
            loss = self._backward_loss(sampled_batch, ks, x0) if training_backward else self._forward_loss(sampled_batch, ks, xN)
            self.manual_backward(loss)
            
            # clip the gradients. first, save the norm for later logging
            norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.clip_gradients(optimizer, self.hparams.max_norm, "norm")
            optimizer.step()
            
            self.curr_num_iters += 1
            
            if not self.cache.is_full():
                break
                
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
    def validation_step(self, batch : Tensor, batch_idx : int, dataloader_idx : int) -> None:
        torch.manual_seed(0)
        is_backward = self.training_backward
        loss_sum = 0.0
        batch = self.encode(batch)
        use_initial_forward_sampling = is_backward and self.DSB_iteration == 1
        
        if (is_backward and dataloader_idx == 0) or (not is_backward and dataloader_idx == 1):
            trajectory = self.sample(batch, forward = is_backward, return_trajectory = True, use_initial_forward_sampling=use_initial_forward_sampling)
            batch_size = trajectory.size(1)
            x0, xN = trajectory[0], trajectory[-1]
            for k in range(0, self.hparams.num_steps):
                k += 1 if is_backward else 0
                ks = self.k_to_tensor(k, batch_size)
                loss = self._backward_loss(trajectory[k], ks, x0) if is_backward else self._forward_loss(trajectory[k], ks, xN)
                loss_sum += loss
            
            loss_avg = loss_sum / self.hparams.num_steps
            self.log(
                self._get_loss_name(backward = is_backward, is_training = False), 
                loss_avg, 
                add_dataloader_idx=False,
                on_step=False,
                )
    
    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        # make the optimizers
        backward_opt = self.partial_optimizer(self.backward_model.parameters())
        forward_opt = self.partial_optimizer(self.forward_model.parameters())
        return backward_opt, forward_opt