from torch import Tensor
import torch
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from functools import partial
from src.lightning_modules.mixins import EncoderDecoderMixin
from src.networks.encoders import BaseEncoderDecoder
from typing import Optional
from src.losses import BaseLoss
from .dsb_scheduler import DSBScheduler, BaggeScheduler, DIRECTIONS, SCHEDULER_TYPES, TensorDict
from collections import deque
import random

class Cache:
    def __init__(
        self,
        max_len: int,
    ):
        self.max_len = max_len
        self.cache = deque(maxlen=max_len)
    
    def add(self, x0: Tensor, x1: Tensor) -> None:
        self.cache.append((x0, x1))
        
    def __len__(self) -> int:
        return len(self.cache)

    def is_full(self) -> bool:
        return len(self.cache) == self.max_len
    
    def sample_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        sample_population = list(self.cache)
        samples = random.sample(sample_population, batch_size)
        x0_batch = torch.stack([s[0] for s in samples], dim=0)
        x1_batch = torch.stack([s[1] for s in samples], dim=0)
        return x0_batch, x1_batch
    
class BaseDSB(BaseLightningModule, EncoderDecoderMixin):
    ema : ExponentialMovingAverage
    
    def __init__(
        self,
        model: Module,
        encoder_decoder: BaseEncoderDecoder,
        pretraining_steps: int, 
        inference_steps: int,
        p_cache: float,
        cache_size: int,
        loss_fn: Optional[BaseLoss] = None,
        optimizer: Optional[partial[Optimizer]] = None,
        lr_scheduler: Optional[dict[str, partial[LRScheduler] | str]] = None,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder', 'loss_fn', 'optimizer', 'lr_scheduler'])
        self.model = torch.compile(model, mode='reduce-overhead', fullgraph=False) if compile else model
        self.encoder_decoder = encoder_decoder
        self.pretraining_steps = pretraining_steps
        self.inference_steps = inference_steps
        self.loss_fn = loss_fn
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        self.cache_size = cache_size
        self.forward_cache = Cache(max_len=cache_size)
        self.backward_cache = Cache(max_len=cache_size)
        self.p_cache = p_cache

    @property
    def is_pretraining(self) -> bool:
        return self.global_step < self.pretraining_steps
        
    def training_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        return self._common_step(batch)
    
    @torch.no_grad()
    def validation_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        with self.fix_validation_seed():
            with self.ema.average_parameters():
                return self._common_step(batch)

    def configure_optimizers(self):
        assert self.partial_optimizer is not None, "Optimizer must be provided during training."
        assert self.partial_lr_scheduler is not None, "Learning rate scheduler must be provided during training."
        
        optim = self.partial_optimizer(self.model.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }
        
    def _common_step(self, batch : tuple[Tensor, Tensor]) -> TensorDict: 
        raise NotImplementedError()
        
    @torch.no_grad()
    def sample(self, x_start : Tensor, direction : DIRECTIONS, scheduler_type : SCHEDULER_TYPES, num_steps : int, return_trajectory : bool, verbose : bool = True) -> Tensor: 
        raise NotImplementedError()

def concat(*xs: Tensor) -> Tensor:
    # concatenate tensors along batch dimension
    return torch.cat(xs, dim=0)

class DSB(BaseDSB):    
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        pretraining_steps: int,
        inference_steps: int,
        cache_size: int,
        p_cache: float,
        loss_fn: Optional[BaseLoss] = None,
        optimizer : Optional[partial[Optimizer]] = None,
        lr_scheduler : Optional[dict[str, partial[LRScheduler] | str]] = None,
        compile: bool = False,
        **scheduler_kwargs,
    ):
        super().__init__(
            model=model,
            encoder_decoder=encoder_decoder,
            pretraining_steps=pretraining_steps,
            inference_steps=inference_steps,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            compile=compile,
            cache_size=cache_size,
            p_cache=p_cache,
        )
        self.scheduler = DSBScheduler(**scheduler_kwargs)
        
    def forward(self, x : Tensor, timesteps : Tensor, conditional : Tensor) -> Tensor:
        return self.model(x, timesteps, conditional)
    
    def training_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        return self._common_step(batch, p_cache=self.p_cache)
        
    def _common_step(self, batch : tuple[Tensor, Tensor], p_cache: float = 0.0) -> TensorDict:        
        x0, x1 = batch
        x0_enc, x1_enc = self.encode_batch(x0, x1)
        
        if self.is_pretraining:
            # split x0 and x1 into backward and forward batches
            x0_b, x0_f = x0_enc.chunk(2, dim=0)
            x1_b, x1_f = x1_enc.chunk(2, dim=0)

        else:
            if self.forward_cache.is_full() and random.random() < p_cache:
                x0_f, x1_f = self.forward_cache.sample_batch(x0_enc.size(0) // 2)
                x0_b, x1_b = self.backward_cache.sample_batch(x0_enc.size(0) // 2)
                x0_f, x1_f = x0_f.to(x0_enc.device), x1_f.to(x1_enc.device)
                x0_b, x1_b = x0_b.to(x0_enc.device), x1_b.to(x1_enc.device)
                
            else:
                x0_b, _ = x0_enc.chunk(2, dim=0)
                x1_f, _ = x1_enc.chunk(2, dim=0)
                
                x1_b = self.sample(x0_b, direction='forward', scheduler_type='cosine', num_steps=self.inference_steps, return_trajectory=False, verbose=False)
                x0_f = self.sample(x1_f, direction='backward', scheduler_type='cosine', num_steps=self.inference_steps, return_trajectory=False, verbose=False)
                
                for i in range(x0_b.size(0)):
                    self.backward_cache.add(x0_b[i].cpu(), x1_b[i].cpu())
                    self.forward_cache.add(x0_f[i].cpu(), x1_f[i].cpu())
                    
                self.log_dict({
                    'cache/forward_cache_size': len(self.forward_cache),
                    'cache/backward_cache_size': len(self.backward_cache),
                    'cache/forward_cache_percent': len(self.forward_cache) / self.cache_size,
                    'cache/backward_cache_percent': len(self.backward_cache) / self.cache_size,
                })        
                
        b_batch = self.scheduler.sample_training_batch(x0_b, x1_b, direction='backward')
        f_batch = self.scheduler.sample_training_batch(x0_f, x1_f, direction='forward')
        
        xt = concat(b_batch['xt'], f_batch['xt'])
        timesteps = concat(b_batch['timesteps'], f_batch['timesteps'])
        conditional = concat(b_batch['conditional'], f_batch['conditional'])
        drift_target = concat(b_batch['drift'], f_batch['drift'])
        drift_pred = self(xt, timesteps, conditional)
        
        loss = self.loss_fn.forward({
            'out': drift_pred,
            'target': drift_target
        })
        
        return loss

    @torch.no_grad()
    def sample(
        self, 
        x_start : Tensor, 
        direction : DIRECTIONS, 
        scheduler_type : SCHEDULER_TYPES, 
        num_steps : int, 
        return_trajectory : bool, 
        verbose : bool = True,
        ) -> Tensor:
        self.model.eval()
        
        batch_size = x_start.shape[0]
        device = x_start.device
        c = self.scheduler.get_conditional(direction, batch_size, device)
        timeschedule = self.scheduler.get_timeschedule(num_steps, scheduler_type, direction)
        x = x_start.clone()
        trajectory = [x]
        for tk_plus_one, tk in tqdm(timeschedule, desc="Sampling...", leave=False, disable=not verbose):
            t = torch.full((batch_size,), tk_plus_one if direction == 'backward' else tk, device=device)
            with self.ema.average_parameters():
                drift = self(x, t, c)
            x = self.scheduler.step(x, drift, tk_plus_one, tk, direction)
            trajectory.append(x)
            
        self.model.train()
            
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        
        return x
    
class BaggeDSB(BaseDSB):
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        pretraining_steps: int,
        inference_steps: int,
        loss_fn: Optional[BaseLoss] = None,
        optimizer : Optional[partial[Optimizer]] = None,
        lr_scheduler : Optional[dict[str, partial[LRScheduler] | str]] = None,
        **scheduler_kwargs,
    ):
        super().__init__(
            model=model,
            encoder_decoder=encoder_decoder,
            pretraining_steps=pretraining_steps,
            inference_steps=inference_steps,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        self.scheduler = BaggeScheduler(**scheduler_kwargs)
        
    def forward(self, x : Tensor, timesteps : Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x, timesteps)
    
    def _common_step(self, batch : tuple[Tensor, Tensor]) -> TensorDict:        
        x0, x1 = batch
        x0_enc, x1_enc = self.encode_batch(x0, x1)
        
        if self.is_pretraining:
            pass

        else:
            x0_b, _ = x0_enc.chunk(2, dim=0)
            x1_f, _ = x1_enc.chunk(2, dim=0)
            x1_b = self.sample(x0_b, direction='forward', scheduler_type='cosine', num_steps=self.inference_steps, return_trajectory=False, verbose=False)
            x0_f = self.sample(x1_f, direction='backward', scheduler_type='cosine', num_steps=self.inference_steps, return_trajectory=False, verbose=False)
            x0_enc = torch.cat([x0_b, x0_f], dim=0)
            x1_enc = torch.cat([x1_b, x1_f], dim=0)
            
        bf_batch = self.scheduler.sample_training_batch(x0_enc, x1_enc)
        flow_target, noise_target = bf_batch['drift'], bf_batch['noise']
        target = torch.cat([flow_target, noise_target], dim=0)
        flow_pred, noise_pred = self.forward(bf_batch['xt'], bf_batch['timesteps'])
        pred = torch.cat([flow_pred, noise_pred], dim=0)
        
        loss = self.loss_fn.forward({
            'out': pred,
            'target': target
        })
        
        return loss

    @torch.no_grad()
    def sample(self, x_start : Tensor, direction : DIRECTIONS, scheduler_type : SCHEDULER_TYPES, num_steps : int, return_trajectory : bool, verbose : bool = True) -> Tensor:
        self.model.eval()
        
        batch_size = x_start.shape[0]
        device = x_start.device
        timeschedule = self.scheduler.get_timeschedule(num_steps, scheduler_type, direction)
        x = x_start.clone()
        trajectory = [x]
        for tk_plus_one, tk in tqdm(timeschedule, desc="Sampling...", leave=False, disable=not verbose):
            t = torch.full((batch_size,), tk_plus_one if direction == 'backward' else tk, device=device)
            with self.ema.average_parameters():
                flow, noise = self(x, t)
            x = self.scheduler.step(x, flow, noise, tk_plus_one, tk, direction)
            trajectory.append(x)
            
        self.model.train()
            
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        
        return x
        

from src.utils import config_from_id, get_ckpt_path
import hydra
        
def load_dsb_model(experiment_id : str) -> DSB:
    config = config_from_id(experiment_id)
    model_config = config['model']
    network = hydra.utils.instantiate(model_config['model'])
    encoder_decoder = hydra.utils.instantiate(model_config['encoder_decoder'])
    ckpt_path = get_ckpt_path(experiment_id, last=False, filename="last.ckpt")
    model = DSB.load_from_checkpoint(ckpt_path, model=network, encoder_decoder=encoder_decoder)
    model.eval()
    return model