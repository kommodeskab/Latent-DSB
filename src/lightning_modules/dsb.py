import torch
from typing import Tuple, Callable, List, Any, Literal
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, ConstantLR
from torch.nn import Module
from pytorch_lightning.utilities import grad_norm
import copy
from src.lightning_modules.utils import Cache, GammaScheduler, get_encoder_decoder
from src.lightning_modules.fm import FM

class DSB(BaseLightningModule):
    def __init__(
        self,
        model : torch.nn.Module,
        encoder_decoder_id : str,
        optimizer : Optimizer,
        scheduler : LRScheduler,
        max_gamma : float,
        min_gamma : float,
        num_steps : int,
        max_iterations : int,
        cache_max_size : int,
        cache_num_iters : int,
        pretrained_forward_model_path : str | None = None,
        pretrained_backward_model_path : str | None = None,
        backward_model : torch.nn.Module | None = None,
        T : int | None = None,
        target : Literal["terminal", "flow"] = "terminal",
        max_dsb_iterations : int | None = 20,
        max_norm : float = float("inf"),
        **kwargs
    ):
        """
        Initializes the StandardDSB model
        """
        
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["model", "optimizer", "scheduler"])
        
        self.training_backward = True
        self.curr_num_iters = 0
        self.DSB_iteration = 1
        
        assert max_gamma >= min_gamma, f"{max_gamma = } must be greater than {min_gamma = }"
        
        self.gamma_scheduler = GammaScheduler(min_gamma, max_gamma, num_steps, T)
        
        # if the backward model is not provided, make it a copy of the forward model 
        self.forward_model = model
        self.backward_model = backward_model if backward_model is not None else copy.deepcopy(model)
        
        # most often, we want to utilize a pretrained diffusion model for the first iterations
        if fp := pretrained_forward_model_path is not None:
            self.forward_diffusion_model = FM.load_from_checkpoint(fp, model = self.forward_model)
            self.forward_diffusion_model.scheduler.set_timesteps(num_steps)
            self.forward_model = self.forward_diffusion_model.model
            
        if bp := pretrained_backward_model_path is not None:
            self.backward_diffusion_model = FM.load_from_checkpoint(bp, model = self.backward_model)
            self.backward_model = self.backward_diffusion_model.model
        
        self.encoder, self.decoder = get_encoder_decoder(encoder_decoder_id)
        
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
                
        self.mse : Callable[[Tensor, Tensor], Tensor] = torch.nn.MSELoss()  
        self.cache = Cache(max_size = self.hparams.cache_max_size)

    def on_fit_start(self) -> None:
        self.trainer.datamodule.training_backward = self.training_backward

    def _has_converged(self) -> bool:
        dsb_iters = self.DSB_iteration
        curr_iters = self.curr_num_iters
        max_iters = self.hparams.max_iterations
        max_dsb_iters = self.hparams.max_dsb_iterations
        return (curr_iters >= max_iters) or (dsb_iters >= max_dsb_iters)

    def on_train_batch_start(self, batch : Tensor, batch_idx : int) -> None:
        """
        We check if the model has converged before each batch
        if the model has converged we do the following:
        - switch the training direction
        - reset the learning rate of the optimizer
        - reset the learning rate scheduler
        
        we also return -1 to skip the rest of the epoch if converged (pytorch lightning behavior)
        
        """

        if self._has_converged():
            self.curr_num_iters = 0
            self.training_backward = not self.training_backward
            self.trainer.datamodule.training_backward = self.training_backward
            self.cache.clear()

            if self.training_backward: 
                # resetting the optimizers and schedulers
                self._reset_optim_and_scheduler()
                self.logger.name
                # save checkpoint under "logs/project/version/checkpoints"
                save_dir = f"logs/{self.logger.name}/{self.logger.version}/checkpoints"
                save_dir = f"{save_dir}/DSB_iteration_{self.DSB_iteration}.ckpt"
                self.trainer.save_checkpoint(save_dir)
                print(f"Saved checkpoint at {save_dir}")

                self.DSB_iteration += 1
                
            return -1
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['DSB_iteration'] = self.DSB_iteration
        
    def on_load_checkpoint(self, checkpoint):
        self.DSB_iteration = checkpoint['DSB_iteration']
        
    def _reset_optim_and_scheduler(self) -> None:
        """
        Resets the optimizer and the scheduler
        """
        optimizers, schedulers = self.configure_optimizers()
        self.trainer.optimizers[0] = optimizers[0]
        self.trainer.optimizers[1] = optimizers[1]
        
        self.trainer.lr_scheduler_configs[0].scheduler = schedulers[0]['scheduler']
        self.trainer.lr_scheduler_configs[1].scheduler = schedulers[1]['scheduler']
    
    def k_to_tensor(self, k : int, size : Tuple[int]) -> Tensor:
        """
        Given k, return a tensor of size 'size' filled with k

        :param int k: the value to fill the tensor with
        :param Tuple[int] size: the size of the tensor

        :return Tensor: the tensor filled with k
        """
        return torch.full((size, ), k, dtype = torch.int64, device = self.device)
    
    def forward_call(self, x : Tensor, k: Tensor) -> Tensor:
        """
        Calls the forward model
        """
        return self.forward_model(x, k)
    
    def backward_call(self, x : Tensor, k : Tensor) -> Tensor:
        """
        Calls the backward model
        """
        return self.backward_model(x, k)
    
    @torch.no_grad()
    def encode(self, x : Tensor) -> Tensor:
        """
        Encodes the input x
        """
        return self.encoder(x)
    
    @torch.no_grad()
    def decode(self, x : Tensor) -> Tensor:
        """
        Decodes the input x
        """
        return self.decoder(x)
        
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
        """
        Compute the loss for the forward model.
        Uses the backward model to 'walk' backwards and optimize the forward model to 'walk' back to the original point
        
        :param Tensor xk: the current point
        :param Tensor ks: the current step
        :param Tensor xN: the end point
        
        :return loss: the loss and the previous point
        """
        pred = self.forward_call(xk, ks)
        if self.hparams.target == "terminal":
            loss = self.mse(pred, xN)
        
        return loss
    
    def _backward_loss(self, xk_plus_one : Tensor, ks_plus_one : int, x0 : Tensor) -> Tensor:
        """
        Compute the loss for the backward model.
        Uses the forward model to 'walk' forward and optimize the backward model to 'walk' back to the original point
        
        :param Tensor xk: the current point
        :param Tensor ks: the current step
        :param Tensor x0: the start point
        
        :return loss: the loss and the next point
        """
        pred = self.backward_call(xk_plus_one, ks_plus_one)
        if self.hparams.target == "terminal":
            loss = self.mse(pred, x0)
            
        return loss
    
    @torch.no_grad()
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False, clamp : bool = False) -> Tensor:
        """
        Given the start point x_start, sample the final point xN / x0 by going forward / backward.
        Also, return the trajectory if return_trajectory is True.
        
        :param Tensor x_start: the start point
        :param bool forward: whether to go forward or backward
        :param bool return_trajectory: whether to return the trajectory
        
        :return Tensor: the final point xN / x0 or the trajectory
        """
        
        # first check if we are using a pretrained diffusion model
        # in that case, we would like to use the pretrained model for sampling in the very first iteration
        if (
            self.DSB_iteration == 1 and 
            forward and
            self.hparams.pretrained_forward_model_path is not None
        ):
            return self.forward_diffusion_model.sample(x_start, return_trajectory, clamp)

        # otherwise, we sample using the diffusion schrödinger bridge
        trajectory = torch.zeros(self.hparams.num_steps + 1, *x_start.size()).to(self.device)
        
        if forward:
            self.forward_model.eval()
            xk = x_start
            trajectory[0] = xk
            for k in range(self.hparams.num_steps): # 0, 1, 2, ..., num_steps - 1
                xk_plus_one = self.go_forward(xk, k)
                trajectory[k + 1] = xk_plus_one
                xk = xk_plus_one

        else:
            self.backward_model.eval()
            xk_plus_one = x_start
            trajectory[-1] = xk_plus_one
            for k in reversed(range(self.hparams.num_steps)): # num_steps - 1, num_steps - 2, ..., 1, 0
                xk = self.go_backward(xk_plus_one, k + 1)
                trajectory[k] = xk
                xk_plus_one = xk
        
        if clamp:
            trajectory = torch.clamp(trajectory, -1, 1)
            xk = torch.clamp(xk, -1, 1)

        return trajectory if return_trajectory else xk
    
    def _get_training_components(self, is_backward : bool) -> Tuple[Module, Optimizer, LRScheduler]:
        """
        Given the model, return the optimizer

        :param bool is_backward: whether to get the backward model or the forward model

        :return Tuple[Module, Optimizer, ExponentialMovingAverage]: the model, optimizer
        """
        backward_optim, forward_optim = self.optimizers()
        backward_scheduler, forward_scheduler = self.lr_schedulers()

        if is_backward:
            return self.backward_model, backward_optim, backward_scheduler
        else:
            return self.forward_model, forward_optim, forward_scheduler

    def _get_loss_name(self, is_backward : bool, is_training : bool):
        iteration = self.DSB_iteration
        direction = "backward" if is_backward else "forward"
        training = "train" if is_training else "val"
        return f"iteration_{iteration}/{direction}_loss/{training}"
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        training_backward = self.training_backward
        
        batch = self.encode(batch)
        trajectory = self.sample(batch, forward = training_backward, return_trajectory = True)
        self.cache.add(trajectory)
        
        model, optimizer, lr_scheduler = self._get_training_components(training_backward)
        model.train()
                
        for _ in range(self.hparams.cache_num_iters):
            self.curr_num_iters += 1

            trajectory = self.cache.sample()
            batch_size = trajectory.size(1)
            
            random_samples = torch.randint(0, batch_size, (batch_size,)).to(self.device)
            random_ks = torch.randint(0, self.hparams.num_steps, (batch_size,)).to(self.device)
            if training_backward:
                random_ks += 1
            sampled_batch = trajectory[random_ks, random_samples]
            x0, xN = trajectory[0][random_samples], trajectory[-1][random_samples]

            # calculate loss and do backward pass
            optimizer.zero_grad()
            # if training backward, then sampled_batch = xk + 1 else sampled_batch = xk
            loss = self._backward_loss(sampled_batch, random_ks, x0) if training_backward else self._forward_loss(sampled_batch, random_ks, xN)
            self.manual_backward(loss)
            
            # raise exception if loss is nan
            if torch.isnan(loss).any():
                raise ValueError(f"Loss is nan:")
            
            if loss > 1.e6:
                raise ValueError(f"Loss is too high: {loss = }")

            # clip the gradients. first, save the norm for later logging
            norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.clip_gradients(optimizer, self.hparams.max_norm, "norm")

            optimizer.step()
            
            # update scheduler
            if isinstance(lr_scheduler, (CosineAnnealingWarmRestarts, ConstantLR)):
                lr_scheduler.step(self.curr_num_iters)
                
        model_name = "backward_model" if training_backward else "forward_model"
        
        self.log_dict({
            self._get_loss_name(is_backward = training_backward, is_training = True): loss,
            f"{model_name}_grad_norm_before_clip": norm
        }, on_step=True)

    @torch.no_grad()
    def validation_step(self, batch : Tensor, batch_idx : int, dataloader_idx : int) -> None:
        self.eval()
        is_backward = self.training_backward
        loss_sum = 0.0
        batch = self.encode(batch)
        
        if (is_backward and dataloader_idx == 0) or (not is_backward and dataloader_idx == 1):
            trajectory = self.sample(batch, forward = is_backward, return_trajectory = True)
            batch_size = trajectory.size(1)
            x0, xN = trajectory[0], trajectory[-1]
            range_ = range(1, self.hparams.num_steps + 1) if is_backward else range(self.hparams.num_steps)
            for k in range_:
                ks = self.k_to_tensor(k, batch_size)
                loss = self._backward_loss(trajectory[k], ks, x0) if is_backward else self._forward_loss(trajectory[k], ks, xN)
                loss_sum += loss
            
            loss_avg = loss_sum / self.hparams.num_steps
            self.log(
                self._get_loss_name(is_backward = is_backward, is_training = False), 
                loss_avg, 
                add_dataloader_idx=False,
                on_step=True,
                )
    
    def on_validation_epoch_end(self) -> None:
        # after validation we want to update the learning rate
        is_backward = self.training_backward
        backward_scheduler, forward_scheduler = self.lr_schedulers()
        scheduler = backward_scheduler if is_backward else forward_scheduler

        metrics = self.trainer.callback_metrics

        # if no losses were computed, return
        if len(metrics) == 0:
            return
        
        val_loss = metrics[self._get_loss_name(is_backward = is_backward, is_training = False)]
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
    
    def configure_optimizers(self):
        # make the optimizers
        backward_opt = self.partial_optimizer(self.backward_model.parameters())
        forward_opt = self.partial_optimizer(self.forward_model.parameters())
        
        lr = backward_opt.param_groups[0]['lr']
        backward_opt.param_groups[0]['lr'] = lr
        forward_opt.param_groups[0]['lr'] = lr

        backward_scheduler = {'scheduler': self.partial_scheduler(backward_opt), 'name': 'lr_scheduler_backward'}
        forward_scheduler = {'scheduler': self.partial_scheduler(forward_opt), 'name': 'lr_scheduler_forward'}
        
        sch = backward_scheduler['scheduler']
        assert isinstance(sch, (CosineAnnealingWarmRestarts, ReduceLROnPlateau, ConstantLR)), f"The scheduler must be either CosineAnnealingWarmRestarts or ReduceLROnPlateau, but got {type(sch)}"
        
        return [backward_opt, forward_opt], [backward_scheduler, forward_scheduler]