from torch import Tensor
import torch
from typing import Any
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.functional import mse_loss
from tqdm import tqdm
from lightning.pytorch.utilities import grad_norm
from torch_ema import ExponentialMovingAverage
from functools import partial
from src.lightning_modules.mixins import EncoderDecoderMixin
from src.networks.encoders import BaseEncoderDecoder
from typing import Optional
from src.losses import BaseLoss
from .dsb_scheduler import DSBScheduler, DIRECTIONS, SCHEDULER_TYPES

class DSB(BaseLightningModule, EncoderDecoderMixin):    
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        loss_fn: Optional[BaseLoss] = None,
        optimizer : Optional[partial[Optimizer]] = None,
        lr_scheduler : Optional[dict[str, partial[LRScheduler] | str]] = None,
        ema_decay : float = 0.999,
        **scheduler_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder', 'loss_fn', 'optimizer', 'lr_scheduler'])
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.loss_fn = loss_fn
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        self.scheduler = DSBScheduler(**scheduler_kwargs)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        self.stop_epoch = False
    
    def on_before_optimizer_step(self, optimizer : Optimizer) -> None:
        if self.global_step % 500 == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def on_before_zero_grad(self, optimizer : Optimizer) -> None:
        self.ema.update()
        
    def on_save_checkpoint(self, checkpoint : dict[str, Any]) -> None:
        checkpoint['ema'] = self.ema.state_dict()
        
    def on_load_checkpoint(self, checkpoint : dict[str, Any]) -> None:
        ema_state_dict = checkpoint['ema']
        self.ema.load_state_dict(ema_state_dict)

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        # dont save encoder_decoder weights since they are frozen during training
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('encoder_decoder.')}
        return state_dict
    
    def load_state_dict(self, state_dict : dict[str, Any], strict = True, assign = False):
        # add encoder_decoder weights back into the state_dict
        encoder_state_dict = self.encoder_decoder.state_dict()
        encoder_state_dict = {f'encoder_decoder.{k}': v for k, v in encoder_state_dict.items()}
        state_dict.update(encoder_state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    
    def to(self, device : torch.device):
        # ema parameters have to be manually moved to the device
        self.ema.to(device)
        return super().to(device)
        
    def forward(self, x : Tensor, timesteps : Tensor, conditional : Tensor) -> Tensor:
        return self.model(x, timesteps, conditional)
    
    def on_train_batch_start(self, batch, batch_idx):
        # pytorch lightning logic for restarting epoch
        if self.stop_epoch:
            self.stop_epoch = False
            return -1
        
    def _common_step(self, batch : tuple[Tensor, Tensor, tuple[str], Tensor]) -> Tensor:        
        x0, x1, direction, is_from_cache = batch
        assert (is_from_cache == is_from_cache[0]).all(), "All tensors in the batch must have the same is_from_cache value."
        if not is_from_cache[0]:
            x0, x1 = self.encode_batch(x0, x1)
        xt, timesteps, conditional, flow = self.scheduler.sample_training_batch(x0, x1, direction)
        model_output = self(xt, timesteps, conditional)
        loss = self.loss_fn({
            'out': model_output, 
            'target': flow
            })
        return loss
        
    def training_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        loss = self._common_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        with self.fix_validation_seed():
            with self.ema.average_parameters():
                loss = self._common_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, x_start : Tensor, direction : DIRECTIONS, scheduler_type : SCHEDULER_TYPES, num_steps : int, return_trajectory : bool, verbose : bool = True) -> Tensor:
        self.model.eval()
        
        batch_size = x_start.shape[0]
        device = x_start.device
        c = self.scheduler.get_conditional(direction, device, batch_size)
        timeschedule = self.scheduler.get_timeschedule(num_steps, scheduler_type, direction)
        x = x_start.clone()
        trajectory = [x]
        for tk_plus_one, tk in tqdm(timeschedule, desc="Sampling...", leave=False, disable=not verbose):
            t = torch.full((batch_size,), tk_plus_one if direction == 'backward' else tk, device=device)
            
            x_input = torch.cat([x, x_start], dim=1) if self.scheduler.condition_on_start else x
                
            with self.ema.average_parameters():
                flow = self(x_input, t, c)
                
            x = self.scheduler.step(x, flow, tk_plus_one, tk, direction)
            trajectory.append(x)
            
        self.model.train()
            
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        
        return x
            
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