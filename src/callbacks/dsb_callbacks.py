from pytorch_lightning import Callback, Trainer
from src.lightning_modules.dsb import DSB, DIRECTIONS
from src.data_modules import DSBDatamodule
from torch.utils.data import DataLoader, Dataset, Subset 
import torch
from tqdm import tqdm
from src.callbacks.utils import get_batch_from_dataset
from src.callbacks.callbacks import plot_images
import matplotlib.pyplot as plt
from torch import Tensor
from torcheval.metrics import FrechetInceptionDistance
from typing import Literal

class DSBCallback(Callback):
    def __init__(
        self,
        cache_size : int,
        validation_cache_size : int,
        cache_batch_size : int,
        refresh_every_n_steps : int,
        num_steps : int = 50,
        steps_before_cache : int | None = None,
        scheduler_type : str = 'cosine',
        plot_every_n_steps : int = 3000,
        cache_num_steps : int | None = None,
        what_media : Literal['image', 'audio'] = 'image',
        enable_checkpointing : bool = True,
    ):
        super().__init__()
        self.cache_size = cache_size
        self.validation_cache_size = validation_cache_size
        self.cache_batch_size = cache_batch_size
        self.steps_before_cache = steps_before_cache if steps_before_cache is not None else float('inf')
        self.refresh_every_n_steps = refresh_every_n_steps
        self.num_steps = num_steps
        self.scheduler_type = scheduler_type
        self.plot_every_n_steps = plot_every_n_steps
        self.cache_num_steps = cache_num_steps or num_steps
        self.what_media = what_media
        self.enable_checkpointing = enable_checkpointing
        
    def save_checkpoint(self, trainer : Trainer, name : str):
        if self.enable_checkpointing:
            filepath = f"logs/{trainer.logger.name}/{trainer.logger.version}/checkpoints/{name}.ckpt"
            print(f"Saving checkpoint to {filepath}...")
            trainer.save_checkpoint(
                filepath=filepath,
            )
    
    def get_dataloader(self, dataset : Dataset, size : int) -> DataLoader:
        indices = torch.randint(0, len(dataset), (size,))
        subset = Subset(dataset, indices)
        
        return DataLoader(
            dataset = subset,
            batch_size = min(self.cache_batch_size, size),
            num_workers = self.num_workers,
            drop_last = False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False, 
        )
    
    def on_train_start(self, trainer : Trainer, pl_module : DSB):
        datamodule : DSBDatamodule = trainer.datamodule
        self.num_workers = datamodule.num_workers
        
        self.cache = datamodule.cache
        self.val_cache = datamodule.val_cache
        self.dataset = datamodule.dataset
        self.valset = datamodule.valset
        
        logger = pl_module.logger
        num_samples = self.validation_cache_size
        
        self.x0_batch = get_batch_from_dataset(self.valset.x0_dataset, num_samples, shuffle=False)
        self.x1_batch = get_batch_from_dataset(self.valset.x1_dataset, num_samples, shuffle=False)

        x0_encoded = pl_module.encode(self.x0_batch[:1].to(pl_module.device)).cpu()
        x1_encoded = pl_module.encode(self.x1_batch[:1].to(pl_module.device)).cpu()
        
        if x0_encoded.dim() == 3:
            x0_encoded = x0_encoded.unsqueeze(1)
            x1_encoded = x1_encoded.unsqueeze(1)
            
        # if the height or width is greater than 1, we plot the encodings as images
        if min(x0_encoded.shape[2:]) > 1:
            for encoding, title in zip([x0_encoded, x1_encoded], ['x0 Encoded', 'x1 Encoded']):
                encoding = encoding.squeeze(0)
                n_channels = encoding.shape[0]
                fig, axs = plt.subplots(1, n_channels, figsize=(n_channels*5, 5))
                # make axs into a list of plt.Axes
                if isinstance(axs, plt.Axes):
                    axs = [axs]
                
                for i in range(n_channels):
                    axs[i].imshow(encoding[i].cpu().numpy())
                    axs[i].set_title(f'Channel {i}')
                    axs[i].axis('off')
                    
                logger.log_image(
                    key = title,
                    images = [fig],
                    step = pl_module.global_step,
                )

        if self.what_media == 'image':
            x0_fig = plot_images(self.x0_batch[:16])
            x1_fig = plot_images(self.x1_batch[:16])
            logger.log_image(
                key = 'Initial Samples',
                images = [x0_fig, x1_fig],
                step = pl_module.global_step,
            )
            
            plt.close('all')
            
        elif self.what_media == 'audio':
            self.sample_rate : int = self.valset.x0_dataset.sample_rate
            x0_audio = [audio.cpu().flatten().numpy() for audio in self.x0_batch[:16]]
            x1_audio = [audio.cpu().flatten().numpy() for audio in self.x1_batch[:16]]
            audios = x0_audio + x1_audio
            caption = ["x0"] * len(x0_audio) + ["x1"] * len(x1_audio)
            
            logger.log_audio(
                key = 'Initial Samples',
                audios = audios,
                sample_rate = [self.sample_rate] * len(audios),
                caption = caption,
                step = pl_module.global_step,
            )
        
    def visualize_samples(self, pl_module : DSB):
        step = pl_module.global_step
        # plot some results        
        device = pl_module.device
        logger = pl_module.logger
        x0_batch, x1_batch = self.x0_batch.to(device), self.x1_batch.to(device)
        x0_encoded, x1_encoded = pl_module.encode(x0_batch), pl_module.encode(x1_batch)
        
        with pl_module.fix_validation_seed():
            forward_trajectory = pl_module.sample(x0_encoded, 'forward', self.scheduler_type, self.num_steps, return_trajectory=True, verbose=True)
            backward_trajectory = pl_module.sample(x1_encoded, 'backward', self.scheduler_type, self.num_steps, return_trajectory=True, verbose=True)

        forward_samples = forward_trajectory[-1, :]
        backward_samples = backward_trajectory[-1, :]
        forward_samples = pl_module.decode(forward_samples)
        backward_samples = pl_module.decode(backward_samples)
        
        # just visualize a single trajectory
        idxs = torch.linspace(0, self.num_steps, 16, dtype=torch.long, device=device)
        forward_trajectory = forward_trajectory[:, 0][idxs]
        backward_trajectory = backward_trajectory[:, 0][idxs]
        forward_trajectory = pl_module.decode(forward_trajectory)
        backward_trajectory = pl_module.decode(backward_trajectory)

        if self.what_media == 'image':
            forward_fd = self.calculate_fid(forward_samples, x1_batch)
            backward_fd = self.calculate_fid(backward_samples, x0_batch)
            
            logger.log_metrics({
                'forward_frechet_distance': forward_fd,
                'backward_frechet_distance': backward_fd,
            }, step=step)
            
            x0_fig = plot_images(forward_samples[:16].cpu())
            x1_fig = plot_images(backward_samples[:16].cpu())
            
            logger.log_image(
                key = 'Samples',
                images = [x0_fig, x1_fig],
                step = step,
            )
        
            # visualize the trajectories
            forward_trajectory_fig = plot_images(forward_trajectory.cpu())
            backward_trajectory_fig = plot_images(backward_trajectory.cpu())
            logger.log_image(
                key = 'Trajectories',
                images = [forward_trajectory_fig, backward_trajectory_fig],
                step = step,
            )
            
            plt.close('all')
            
        elif self.what_media == 'audio':
            x0_audio = [audio.cpu().flatten().numpy() for audio in backward_samples[:16]]
            x1_audio = [audio.cpu().flatten().numpy() for audio in forward_samples[:16]]
            audios = x0_audio + x1_audio
            caption = ["x0"] * len(x0_audio) + ["x1"] * len(x1_audio)
            
            logger.log_audio(
                key = 'Samples',
                audios = audios,
                sample_rate = [self.sample_rate] * len(audios),
                caption = caption,
                step = step,
            )
            
    def refresh_cache(self, pl_module : DSB, direction : DIRECTIONS, training : bool) -> None:
        """Refresh the cache for the given direction and training mode.

        Args:
            pl_module (DSB): The DSB module.
            direction (DIRECTIONS): The direction of the cache ('forward' or 'backward'). Notice: this direction is the OPPOSITE of the sampling direction.
            training (bool): Which cache: train if True, validation if False.
        """
        
        if training and direction == 'forward': cache, dataset = self.cache, self.dataset.x1_dataset
        if training and direction == 'backward': cache, dataset = self.cache, self.dataset.x0_dataset

        if not training and direction == 'forward': cache, dataset = self.val_cache, self.valset.x1_dataset
        if not training and direction == 'backward': cache, dataset = self.val_cache, self.valset.x0_dataset
        
        device = pl_module.device
        num_samples = self.cache_size if training else self.validation_cache_size
        
        dataloader = self.get_dataloader(dataset, num_samples)
        
        # careful: backward cache is filled with x1 samples when sampling forward and vice versa
        sampling_direction = 'backward' if direction == 'forward' else 'forward'

        for x_start in tqdm(dataloader, desc=f"Sampling {sampling_direction} for {'training cache' if training else 'validation cache'}...", leave=False):
            x_start : torch.Tensor
            x_start = x_start.to(device)
            x_start = pl_module.encode(x_start)
        
            x_end = pl_module.sample(
                x_start = x_start, 
                direction = sampling_direction, 
                scheduler_type = self.scheduler_type,
                num_steps = self.cache_num_steps,
                return_trajectory = False,
                verbose = False,
            )
                        
            # put to cpu and convert to float16 to save memory
            x_start = x_start.cpu().to(torch.float16)
            x_end = x_end.cpu().to(torch.float16)

            if sampling_direction == 'forward':
                x0, x1 = x_start, x_end
            elif sampling_direction == 'backward':
                x0, x1 = x_end, x_start
            
            cache.add(x0, x1, direction)
        
    def visualize_caches(self, pl_module : DSB):
        step = pl_module.global_step
        caches = [self.cache, self.val_cache]
        titles = ['Training Cache', 'Validation Cache']
        
        for cache, title in zip(caches, titles):
            batches = get_batch_from_dataset(cache, 16, shuffle=False)
            x0_samples, x1_samples, direction, is_from_cache = batches
            x0_samples, x1_samples = x0_samples.float(), x1_samples.float()
            
            device = pl_module.device
            x0_samples = pl_module.decode(x0_samples.to(device)).cpu()
            x1_samples = pl_module.decode(x1_samples.to(device)).cpu()
            
            if self.what_media == 'image':
                x0_fig = plot_images(x0_samples)
                x1_fig = plot_images(x1_samples)
                logger = pl_module.logger
                logger.log_image(
                    key = title,
                    images = [x0_fig, x1_fig],
                    step = step,
                )
                plt.close('all')
                
            elif self.what_media == 'audio':
                pass
            
    def on_validation_end(self, trainer : Trainer, pl_module : DSB):
        self.save_checkpoint(trainer, 'last')
        
    def on_train_end(self, trainer : Trainer, pl_module : DSB):
        self.save_checkpoint(trainer, 'last')
        
    def on_train_batch_end(self, trainer : Trainer, pl_module : DSB, outputs, batch, batch_idx):
        step = pl_module.global_step + 1 # 1, 2, 3, ...
        
        if step % self.plot_every_n_steps == 0:
            # visualize samples
            self.visualize_samples(pl_module)
        
        # check if we should refresh the cache
        steps_since_cache = step - self.steps_before_cache - 1
        
        if steps_since_cache < 0:
            # not enough steps have passed to start caching
            return
        
        if steps_since_cache == 0:
            # done with pretraining and starting cache
            # we are going to save a checkpoint with current weights
            self.save_checkpoint(trainer, 'pretraining')
        
        if (steps_since_cache % self.refresh_every_n_steps == 0) or self.cache.is_empty():
            self.cache.clear()
            self.val_cache.clear()
            
            self.refresh_cache(pl_module, 'forward', training=True)
            self.refresh_cache(pl_module, 'backward', training=True)
            self.refresh_cache(pl_module, 'forward', training=False)
            self.refresh_cache(pl_module, 'backward', training=False)
            
            pl_module.logger.log_metrics({
                'Train cache': len(self.cache.cache),
                'Validation cache': len(self.val_cache.cache),
            }, step = pl_module.global_step)
            
            self.visualize_caches(pl_module)
            
            pl_module.stop_epoch = True
            
    @torch.no_grad()
    def calculate_fid(self, x : Tensor, y : Tensor) -> Tensor:
        if not hasattr(self, 'fid_metric'):
            self.fid_metric = FrechetInceptionDistance(device=x.device)
        
        # normalize images to [0, 1] range
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = x.clamp(0, 1)
        y = y.clamp(0, 1)
            
        self.fid_metric.update(x, is_real=True)
        self.fid_metric.update(y, is_real=False)
        fid_score = self.fid_metric.compute()
        self.fid_metric.reset()
        return fid_score
        