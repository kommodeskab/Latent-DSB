from torch import Tensor
import torch
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from functools import partial
from src.lightning_modules.mixins import EncoderDecoderMixin
from src.networks.encoders import BaseEncoderDecoder
from typing import Optional
from src.losses import BaseLoss
from .dsb_scheduler import DSBScheduler, DIRECTIONS, SCHEDULER_TYPES, TensorDict
from src.networks.encoders import BaseEncoderDecoder
from src.types import TensorDict
from src.dataset.dsb_dataset import Cache, DSBDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import logging
import math

logger = logging.getLogger(__name__)

def concat(*xs: Tensor) -> Tensor:
    # concatenate tensors along batch dimension
    return torch.cat(xs, dim=0)

class DSB(BaseLightningModule, EncoderDecoderMixin):    
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        pretraining_steps: int,
        inference_steps: int,
        cache_batch_size: int,
        populate_every_n_steps: int,
        scheduler: DSBScheduler,
        loss_fn: Optional[BaseLoss] = None,
        optimizer : Optional[partial[Optimizer]] = None,
        lr_scheduler : Optional[dict[str, partial[LRScheduler] | str]] = None,
    ):
        super().__init__(optimizer, lr_scheduler)
        self.save_hyperparameters(ignore=['model', 'encoder_decoder', 'loss_fn', 'optimizer', 'lr_scheduler'])
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.pretraining_steps = pretraining_steps
        self.inference_steps = inference_steps
        self.cache_batch_size = cache_batch_size
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.populate_every_n_steps = populate_every_n_steps
            
    @property
    def trainset(self) -> DSBDataset:
        return self.datamodule.train_dataset # type: ignore
    
    @property
    def valset(self) -> DSBDataset:
        return self.datamodule.val_dataset # type: ignore
        
    @property
    def pretraining(self) -> bool:
        return self.global_step < self.pretraining_steps
    
    def forward(self, x : Tensor, timesteps : Tensor, conditional : Tensor) -> Tensor:
        return self.model(x, timesteps, conditional)
    
    def populate_cache(self, cache: Cache, dataset: Dataset, direction: DIRECTIONS):
        n_batches = math.ceil(cache.maxlen / self.cache_batch_size / self.trainer.world_size)
        
        for _ in range(n_batches):
            loader = DataLoader(dataset, batch_size=self.cache_batch_size, shuffle=True, drop_last=True)
            x_start: Tensor = next(iter(loader)).to(self.device)
            x_start = self.encode(x_start)
                    
            x_end = self.sample(x_start=x_start, direction=direction, scheduler_type='cosine', num_steps = self.inference_steps, return_trajectory=False, verbose=True).cpu()
            (x0, x1) = (x_start, x_end) if direction == 'forward' else (x_end, x_start)
            
            for i in range(x_start.shape[0]):
                cache.add(x0[i], x1[i])
                
    def populate_all_caches(self):
        self.trainer.strategy.barrier()
        logger.info("Populating caches...")
        self.populate_cache(self.trainset.backward_cache, self.trainset.x0_dataset, direction='forward')
        self.populate_cache(self.trainset.forward_cache, self.trainset.x1_dataset, direction='backward')
        self.trainer.strategy.barrier()
        
    @property
    def is_pretraining(self):
        return self.global_step < self.pretraining_steps

    @property
    def is_finetuning(self):
        return not self.is_pretraining
        
    def on_train_batch_start(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        if self.is_finetuning:
            if not self.trainset.caches_are_full:
                # always populate caches if the caches are empty while finetuning
                self.populate_all_caches()
                return -1
                
            elif (self.global_step % self.populate_every_n_steps) == 0 and self.current_epoch > 0:
                # periodically populate caches during finetuning
                self.populate_all_caches()
        
    def common_step(self, batch : tuple[Tensor, ...], batch_idx: int) -> TensorDict:        
        x0_f, x1_f, x0_b, x1_b = batch
        
        if self.is_pretraining or not self.training:
            x0_f, x1_f, x0_b, x1_b = self.encode_batch(x0_f, x1_f, x0_b, x1_b)

        b_batch = self.scheduler.sample_training_batch(x0_b, x1_b, direction='backward')
        f_batch = self.scheduler.sample_training_batch(x0_f, x1_f, direction='forward')
        
        xt = concat(b_batch['xt'], f_batch['xt'])
        timesteps = concat(b_batch['timesteps'], f_batch['timesteps'])
        conditional = concat(b_batch['conditional'], f_batch['conditional'])
        target = concat(b_batch['target'], f_batch['target'])
        pred = self(xt, timesteps, conditional)
        
        loss = self.loss_fn.forward({
            'out': pred,
            'target': target
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
            x_in = torch.cat([x, x_start], dim=1) if self.scheduler.condition_on_start else x
            model_output = self.forward(x_in, t, c)
            x = self.scheduler.step(x, model_output, tk_plus_one, tk, direction)
            trajectory.append(x)
            
        self.model.train()
            
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        
        return x