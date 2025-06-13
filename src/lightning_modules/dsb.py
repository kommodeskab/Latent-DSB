import torch
from typing import Tuple, Literal
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from src.networks.encoders import BaseEncoderDecoder
from torch.optim import Optimizer
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import mse_loss
from torch.nn import Module
from .mixins import EncoderDecoderMixin
from .schedulers import DSBScheduler, NOISE_TYPES, TARGETS
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset, DataLoader
from src.data_modules.base_dm import split_dataset
import math
import gc
import os

class DSBCacheDataset(Dataset):
    def __init__(self, max_size : int, num_cache_iterations : int):
        super().__init__()
        self.cache = []
        self.num_cache_iterations = num_cache_iterations
        self.max_size = max_size
        
    def __len__(self):
        return len(self.cache) * self.num_cache_iterations
    
    def add(self, trajectory : Tensor):
        batch_size = trajectory.size(1)
        for i in range(batch_size):
            self.cache.append(trajectory[:, i])
            
    def clear(self):
        # remove all tensors from memory
        self.cache.clear()
        gc.collect()
        
    def is_full(self):
        return len(self.cache) >= self.max_size
    
    def is_empty(self):
        return len(self.cache) == 0
            
    def __getitem__(self, idx : int) -> Tensor:
        idx = idx % len(self.cache)
        return self.cache[idx]
        

class DSB(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        backward_model : Module,
        forward_model : Module,
        encoder_decoder : BaseEncoderDecoder,
        num_iterations : int,
        effective_batch_size : int,
        cache_max_size : int,
        num_cache_iterations : int,
        cache_generator_batch_size : int = 128,
        num_workers : int = 4,
        num_timesteps : int = 100,
        first_iteration : Literal['network', 'network-straight', 'noise', 'pretrained'] = 'network',
        gamma_min : float | None = None,
        gamma_max : float | None = None,
        optimizer : Optimizer | None = None,
        target : TARGETS = "terminal",
        max_norm : float = 1.0,
        ema_decay : float = 0.999,
        max_DSB_iterations : int = float('inf'),
        pretrained_target : TARGETS | None = None,
        x0_dataset : Dataset | None = None,
        x1_dataset : Dataset | None = None,
        x0_dataset_val : Dataset | None = None,
        x1_dataset_val : Dataset | None = None,
        num_iterations_mult : int = 1,
        minimum_num_iterations : int = 0,
    ):
        super().__init__()
        assert first_iteration in ["network", "network-straight", "noise", "pretrained"], "Invalid first_iteration"
        
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["backward_model", "forward_model", "encoder_decoder", "optimizer" "x0_dataset", "x1_dataset", "x0_dataset_val", "x1_dataset_val"])
        
        self.training_backward = True
        self.curr_num_iters = 0
        self.DSB_iteration = 1
        self.first_iteration = first_iteration
        
        self.scheduler = DSBScheduler(num_timesteps, gamma_min, gamma_max, target)
        
        self.forward_model = forward_model
        self.backward_model = backward_model
        
        if first_iteration == "pretrained":
            assert pretrained_target is not None, "Pretrained target must be provided"
            self.first_iteration_scheduler = DSBScheduler(num_timesteps, gamma_min, gamma_max, pretrained_target)

        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        
        self.forward_ema = ExponentialMovingAverage(self.forward_model.parameters(), decay=ema_decay)
        self.backward_ema = ExponentialMovingAverage(self.backward_model.parameters(), decay=ema_decay)
        
        if x0_dataset is not None:
            self.original_dataset = x0_dataset # for accessing hyperparameters
            self.x0_dataset, self.x0_dataset_val = split_dataset(x0_dataset, x0_dataset_val, 0.95)
            self.x1_dataset, self.x1_dataset_val = split_dataset(x1_dataset, x1_dataset_val, 0.95)
        
        self.max_norm = max_norm
        self.effective_batch_size = effective_batch_size
        self.cache_generator_batch_size = cache_generator_batch_size
        self.num_workers = num_workers
        self.num_iterations = num_iterations
        self.cache_max_size = cache_max_size
        self.num_cache_iterations = num_cache_iterations
        self.max_DSB_iterations = max_DSB_iterations
        self.num_iterations_mult = num_iterations_mult
        self.minimum_num_iterations = minimum_num_iterations 
            
    def num_allowed_iterations(self, dsb_iteration : int) -> int:
        # returns the allowed number of training iterations for the given DSB iteration
        allowed_iterations = self.num_iterations * self.num_iterations_mult ** (dsb_iteration - 1)
        allowed_iterations = max(allowed_iterations, self.minimum_num_iterations)
        return int(allowed_iterations)
        
    def fill_up_cache(self, cache : DSBCacheDataset, sample_forward : bool, mode : Literal['training', 'validation']) -> Dataset:    
        if mode == 'training':
            dataset = self.x0_dataset if sample_forward else self.x1_dataset
        elif mode == 'validation':
            dataset = self.x0_dataset_val if sample_forward else self.x1_dataset_val
        
        def infinite_dataloader(dataset : Dataset, batch_size : int, num_workers : int):
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=False,
            )
            
            while True:
                for data in dataloader:
                    yield data
        
        num_batches = math.ceil(cache.max_size / self.cache_generator_batch_size)
        batch_size = cache.max_size // num_batches
        dataloader = infinite_dataloader(dataset, batch_size, self.num_workers)
        
        for i, x_start in enumerate(dataloader, start=1):
            dtype = self.get_precision_dtype()
            x_start : Tensor
            x_start = x_start.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=dtype): # disable cache to save memory
                x_start = self.encode(x_start)
                trajectory = self.sample(x_start, forward=sample_forward, return_trajectory=True, noise='training', show_progress=True)
                trajectory = trajectory.cpu().to(dtype=dtype)
                cache.add(trajectory)
            
            if i >= num_batches:
                break
            
        del dataloader
                
    def train_dataloader(self):
        rank = self.global_rank
        self.train_cache[rank].clear()
        self.fill_up_cache(self.train_cache[rank], self.training_backward, 'training')
                
        return DataLoader(
            dataset = self.train_cache[rank],
            batch_size = self.effective_batch_size,
            shuffle = True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        
        
    def val_dataloader(self):
        rank = self.global_rank
        
        if self.val_cache[rank].is_empty():
            # only make a validation dataset if we don't have one already
            self.fill_up_cache(self.val_cache[rank], self.training_backward, mode='validation')
        
        return DataLoader(
            dataset = self.val_cache[rank],
            batch_size = self.effective_batch_size,
            shuffle = False,
            drop_last = True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        
    def get_precision_dtype(self):
        dtype_str = self.trainer.precision
        precision_dict = {
            'bf16-mixed': torch.bfloat16,
            '16-mixed': torch.float16,
            '32-true': torch.float32,
            '16-true': torch.float16,
        }
        assert dtype_str in precision_dict, f"Unknown precision {dtype_str}. Please use either {', '.join(precision_dict.keys())}."
        return precision_dict[dtype_str]
    
    def to(self, device : str | torch.device) -> None:
        self.forward_ema.to(device)
        self.backward_ema.to(device)
        self.encoder_decoder.to(device)
        return super().to(device)
        
    def on_fit_start(self) -> None:
        # for good measure, print the amount of allowed training iterations for each DSB iteration
        print(f"Allowed training iterations for each DSB iteration:")
        allowed_iterations = [self.num_allowed_iterations(i) for i in range(1, self.max_DSB_iterations + 1)]
        for i, iterations in enumerate(allowed_iterations, start=1):
            print(f"DSB iteration {i}: {iterations} iterations")
            
        print(f"Total number of training iterations: {2 * sum(allowed_iterations)}")
        
        world_size = self.trainer.world_size
        train_cache_size = self.cache_max_size // world_size
        # fix validation cache size to be 32 samples
        val_cache_size = self.cache_generator_batch_size // world_size
        print(f"World size: {world_size}, cache size per GPU: {train_cache_size} (training) {val_cache_size} (validation).")
        
        self.train_cache = [DSBCacheDataset(train_cache_size, self.num_cache_iterations) for _ in range(world_size)]
        self.val_cache = [DSBCacheDataset(val_cache_size, 10) for _ in range(world_size)]
        
    def on_train_batch_start(self, batch : Tensor, batch_idx : int) -> None:
        self.curr_num_iters += 1
        
        # calculate how many training iterations we are allowed to do for this DSB iteration
        allowed_iterations = self.num_allowed_iterations(self.DSB_iteration)
        
        if self.curr_num_iters >= allowed_iterations:
            self.curr_num_iters = 0
            self.training_backward = not self.training_backward
            
            # clear the validation cache dataset since each iteration is independent
            for cache in self.val_cache:
                cache.clear()
            
            self.forward_ema.copy_to()
            self.backward_ema.copy_to()

            if self.training_backward: 
                self._reset_optimizers()
                self.DSB_iteration += 1
                
            if self.DSB_iteration > self.max_DSB_iterations:
                # stop training
                print(f"Stopping training at DSB iteration {self.DSB_iteration}.")
                self.trainer.should_stop = True
                
            return -1
        
    def save_checkpoint(self, dsb_iteration : int) -> None:
        save_dir = f"logs/{self.logger.name}/{self.logger.version}/checkpoints/DSB_iteration_{dsb_iteration}.ckpt"
        print(f"Saving checkpoint to {save_dir}")
        self.trainer.save_checkpoint(save_dir)
        
    def on_validation_end(self):
        # save checkpoint under "logs/project/version/checkpoints"
        self.save_checkpoint(self.DSB_iteration)
        
    def on_train_start(self):
        if self.DSB_iteration == 1 and self.training_backward:
            # this checkpoint can be used for validating the "initial" model
            self.save_checkpoint(0) 
        
    def state_dict(self):
        _, backward_ema = self.get_model(backward=True)
        _, forward_ema = self.get_model(backward=False)
        with forward_ema.average_parameters():
            with backward_ema.average_parameters():
                ema_state_dict = super().state_dict()
                
        return ema_state_dict
        
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
        noise_factor : float = 1.0,
    ) -> Tensor:
        model, ema = self.get_model(backward = not forward)
        model.eval()
        xk = x_start.clone()
        trajectory = [xk]
        batch_size = xk.size(0)
        
        generator = reversed(list(enumerate(scheduler.timesteps, start=1)))
        for k, t in tqdm(generator, desc='Sampling', disable=not show_progress, leave=False, total=len(scheduler.timesteps)):
            ts = torch.full((batch_size,), t, dtype=torch.int16, device=xk.device)
            with ema.average_parameters():
                model_output = model(xk, ts)
            xk = scheduler.step(xk, k, model_output, noise, noise_factor)
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
        noise_factor : float = 1.0,
    ) -> Tensor:
        if (self.training_backward and self.DSB_iteration == 1 and forward): # when sampling forward in the first iteration
            if self.first_iteration == 'noise':
                x_end = torch.randn_like(x_start)
                return self.scheduler.deterministic_sample(x_start, x_end, return_trajectory, noise, noise_factor)
            
            elif self.first_iteration == 'pretrained':
                return self._sample(x_start, self.first_iteration_scheduler, forward, return_trajectory, show_progress, noise, noise_factor)
        
        if self.first_iteration == 'network-straight':
            x_end = self._sample(x_start, self.scheduler, forward, False, show_progress, 'inference', noise_factor)
            return self.scheduler.deterministic_sample(x_start, x_end, return_trajectory, noise, noise_factor)
        
        return self._sample(x_start, self.scheduler, forward, return_trajectory, show_progress, noise, noise_factor)
    
    def chunk_sample(self, x_start : Tensor, forward : bool, chunk_size : int, return_trajectory : bool = False, show_progress : bool = False, noise : NOISE_TYPES = 'inference') -> Tensor:
        # split x_start into chunks of size chunk_size
        chunks = torch.split(x_start, chunk_size)
        outputs = []
        for chunk in chunks:
            output = self.sample(chunk, forward, return_trajectory, show_progress, noise)
            outputs.append(output)
            
        if return_trajectory:
            return torch.cat(outputs, dim=1)
        else:
            return torch.cat(outputs, dim=0)
        
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
        # permute the first two dimensions of the batch
        batch = batch.permute(1, 0, *range(2, len(batch.shape)))
        
        model, ema = self.get_model(backward = self.training_backward)
        if not model.training:
            model.train()
        
        optimizer = self.get_optimizer(backward = self.training_backward)
        xt, timesteps, target = self.scheduler.sample_batch(batch)
        model_output = model(xt, timesteps)
        loss = mse_loss(model_output, target)
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(optimizer, self.max_norm, "norm")
        optimizer.step()
        ema.update()
        
        self.log_dict({
            self._get_loss_name(backward = self.training_backward, is_training = True): loss,
            "curr_num_iters": self.curr_num_iters,
            "DSB_iteration": self.DSB_iteration,
            "training_backward": self.training_backward,
            f"cache_size_{self.global_rank}": len(self.train_cache[self.global_rank].cache),
        }, on_step=True, prog_bar=True, sync_dist=True)
        
    def on_before_optimizer_step(self, optimizer):
        model, _ = self.get_model(backward=self.training_backward)
        if self.global_step % 200 == 0: # only log the norm rarely, it is time consuming
            norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.log(f"{'backward' if self.training_backward else 'forward'}_norm", norm, prog_bar=True)

    @torch.no_grad()
    def validation_step(self, batch : Tensor, batch_idx : int) -> None:
        batch = batch.permute(1, 0, *range(2, len(batch.shape)))
        
        with self.fix_validation_seed():
            model, ema = self.get_model(backward = self.training_backward)
            if model.training:
                model.eval()
                
            xt, timesteps, target = self.scheduler.sample_batch(batch)
            with ema.average_parameters():
                model_output = model(xt, timesteps)
                
            loss = mse_loss(model_output, target)
            
            self.log_dict({
                self._get_loss_name(backward = self.training_backward, is_training = False): loss,
                "curr_num_iters": self.curr_num_iters,
            }, on_step=False, prog_bar=True, sync_dist=True)        
    
    def configure_optimizers(self) -> Tuple[list[Optimizer], list[dict]]:
        # make the optimizers
        backward_opt : Optimizer = self.partial_optimizer(self.backward_model.parameters())
        forward_opt : Optimizer = self.partial_optimizer(self.forward_model.parameters())
        return backward_opt, forward_opt

from src.utils import config_from_id, get_ckpt_path
import hydra

def load_dsb_model(experiment_id : str, dsb_iteration : int) -> DSB:
    config = config_from_id(experiment_id)
    model_config = config['model']
    forward_model = hydra.utils.instantiate(model_config['forward_model'])
    backward_model = hydra.utils.instantiate(model_config['backward_model'])
    encoder_decoder = hydra.utils.instantiate(model_config['encoder_decoder'])
    ckpt_path = get_ckpt_path(experiment_id, last=False, filename=f"DSB_iteration_{dsb_iteration}.ckpt")
    model = DSB.load_from_checkpoint(ckpt_path, forward_model=forward_model, backward_model=backward_model, encoder_decoder=encoder_decoder, x0_dataset=None, x1_dataset=None, x0_dataset_val=None, x1_dataset_val=None)
    if experiment_id == '180425125453':
        model.encoder_decoder.off_set = 0
    model = model.eval()
    return model

def load_dsb_datasets(experiment_id : str) -> Tuple[Dataset, Dataset]:
    config = config_from_id(experiment_id)
    model_config = config['model']
    # return the validation datasets
    x0_config = model_config['x0_dataset_val']
    x1_config = model_config['x1_dataset_val']
    x0_dataset = hydra.utils.instantiate(x0_config)
    x1_dataset = hydra.utils.instantiate(x1_config)
    return x0_dataset, x1_dataset

def get_dsb_iterations(experiment_id : str) -> list[int]:
    """
    Given a list of DSB iterations, return the list of DSB iterations that exist in the logs.
    """
    dsb_iterations = []
    # no need to check more than 20 iterations, cause we don't have that many
    for i in range(0, 20):
        if os.path.exists(f"logs/dsb/{experiment_id}/checkpoints/DSB_iteration_{i}.ckpt"):
            dsb_iterations.append(i)
    return dsb_iterations
    
class PretrainedDSBModel:
    def __new__(cls, experiment_id : str, dsb_iteration : int) -> DSB: return load_dsb_model(experiment_id, dsb_iteration)