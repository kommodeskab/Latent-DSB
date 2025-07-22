from pytorch_lightning import Callback, Trainer
from src.lightning_modules.esdsb import ESDSB
from src.data_modules.esdsb_datamodule import ESDSBDatamodule
from torch.utils.data import DataLoader, Dataset, Subset 
import torch
from tqdm import tqdm
from src.callbacks.utils import get_batch_from_dataset
from src.callbacks.callbacks import plot_images
import matplotlib.pyplot as plt
from torch import Tensor

class ESDSBCallback(Callback):
    def __init__(
        self,
        cache_size : int,
        cache_batch_size : int,
        refresh_every_n_steps : int,
        num_steps : int = 50,
        steps_before_cache : int | None = None,
        scheduler_type : str = 'cosine',
        plot_every_n_steps : int = 3000,
    ):
        super().__init__()
        self.cache_size = cache_size
        self.cache_batch_size = cache_batch_size
        self.steps_before_cache = steps_before_cache if steps_before_cache is not None else float('inf')
        self.refresh_every_n_steps = refresh_every_n_steps
        self.num_steps = num_steps
        self.scheduler_type = scheduler_type
        self.plot_every_n_steps = plot_every_n_steps
        
    def get_subset(self, dataset : Dataset) -> Subset:
        indices = torch.randint(0, len(dataset), (self.cache_size,))
        return Subset(dataset, indices)
    
    def get_dataloader(self, dataset : Dataset) -> DataLoader:
        subset = self.get_subset(dataset)
        
        return DataLoader(
            dataset = subset,
            batch_size = self.cache_batch_size,
            num_workers = self.num_workers,
            drop_last = False,
        )
    
    def on_train_start(self, trainer : Trainer, pl_module : ESDSB):
        datamodule : ESDSBDatamodule = trainer.datamodule
        self.num_workers = datamodule.num_workers
        self.dsbdataset = datamodule.dsbdataset
        self.x0_dataset = datamodule.dsbdataset.x0_dataset
        self.x1_dataset = datamodule.dsbdataset.x1_dataset
        
        self.x0_batch = get_batch_from_dataset(self.x0_dataset, 128, shuffle=False)
        self.x1_batch = get_batch_from_dataset(self.x1_dataset, 128, shuffle=False)
                
        logger = pl_module.logger
        x0_fig = plot_images(self.x0_batch[:16])
        x1_fig = plot_images(self.x1_batch[:16])
        logger.log_image(
            key = 'Initial Samples',
            images = [x0_fig, x1_fig],
            step = 0,
        )
        
        # visualize training steps, only for the forward direction
        directions = tuple(['backward' for _ in range(16)])
        timesteps = torch.linspace(1e-5, 1, 16, dtype=torch.float32)
        x0 = self.x0_batch[0].repeat(16, 1, 1, 1)
        x1 = self.x1_batch[0].repeat(16, 1, 1, 1)
        xt, timesteps, conditional, flow = pl_module.scheduler.sample_training_batch(x0, x1, directions, timesteps)
        # visualize xt and flow
        xt_fig = plot_images(xt)
        flow_fig = plot_images(flow)
        logger.log_image(
            key = 'Training Step Visualization',
            images = [xt_fig, flow_fig],
            step = 0,
        )
        self.training_step_xt = xt
        self.training_timesteps = timesteps
        self.training_conditional = conditional
        
        plt.close('all')
            
    def on_train_batch_end(self, trainer : Trainer, pl_module : ESDSB, *args, **kwargs):   
        step = pl_module.global_step + 1
             
        if step % self.plot_every_n_steps == 0:
            # plot some results        
            device = pl_module.device
            logger = pl_module.logger
            x0_batch, x1_batch = self.x0_batch.to(device), self.x1_batch.to(device)
            with pl_module.fix_validation_seed():
                forward_trajectory = pl_module.sample(x0_batch, 'forward', self.scheduler_type, self.num_steps, return_trajectory=True, verbose=True)
                backward_trajectory = pl_module.sample(x1_batch, 'backward', self.scheduler_type, self.num_steps, return_trajectory=True, verbose=True)
            
            forward_trajectory, backward_trajectory = forward_trajectory.cpu(), backward_trajectory.cpu()
            
            forward_samples = forward_trajectory[:, -1]
            backward_samples = backward_trajectory[:, -1]
            
            x0_fig = plot_images(forward_samples[:16])
            x1_fig = plot_images(backward_samples[:16])
            logger.log_image(
                key = 'Samples',
                images = [x0_fig, x1_fig],
                step = step,
            )
            
            # calculate mmd between generated samples and real samples
            forward_mmd = self.calculate_mmd(forward_samples, x1_batch.cpu())
            backward_mmd = self.calculate_mmd(backward_samples, x0_batch.cpu())
            logger.log_metrics({
                'forward_mmd': forward_mmd.item(),
                'backward_mmd': backward_mmd.item(),
            }, step=step)
            
            # visualize the trajectories
            idxs = torch.linspace(0, self.num_steps, 16, dtype=torch.long)
            forward_single_trajectory = forward_trajectory[0, :][idxs]
            backward_single_trajectory = backward_trajectory[0, :][idxs]
            forward_trajectory_fig = plot_images(forward_single_trajectory)
            backward_trajectory_fig = plot_images(backward_single_trajectory)
            logger.log_image(
                key = 'Trajectories',
                images = [forward_trajectory_fig, backward_trajectory_fig],
                step = step,
            )
            
            plt.close('all')
        
        # check if we should refresh the cache
        steps_since_cache = step - self.steps_before_cache - 1
        
        if steps_since_cache < 0:
            # not enough steps have passed to start caching
            return
        
        if steps_since_cache % self.refresh_every_n_steps == 0:
            # clear the cache
            self.dsbdataset.forward_cache[:] = []
            self.dsbdataset.backward_cache[:] = []
            
            device = pl_module.device
            
            for direction in ['forward', 'backward']:
                dataset = self.x0_dataset if direction == 'forward' else self.x1_dataset
                dataloader = self.get_dataloader(dataset)
                
                for x_start in tqdm(dataloader, desc=f"Sampling {direction} cache...", leave=False):
                    x_start : Tensor
                    x_start = x_start.to(device)
                    
                    x_end = pl_module.sample(
                        x_start = x_start, 
                        direction = direction, 
                        scheduler_type = self.scheduler_type,
                        num_steps = self.num_steps,
                        return_trajectory = False,
                        verbose = False,
                    )
                    
                    # put to cpu and convert to float16 to save memory
                    x_start = x_start.cpu().to(torch.float16)
                    x_end = x_end.cpu().to(torch.float16)
                    
                    if direction == 'forward':
                        x0, x1 = x_start, x_end
                    elif direction == 'backward':
                        x0, x1 = x_end, x_start
                        
                    # CAREFUL HERE
                    # when sampling forward, we should add to the backward cache and vice versa
                    # since the backward model is trained on forward samples
                    what_cache = 'backward' if direction == 'forward' else 'forward'
                    self.dsbdataset.add_to_cache(x0, x1, what_cache)
            
            # visualize the cache
            caches = [self.dsbdataset.forward_cache, self.dsbdataset.backward_cache]
            titles = ['Forward Cache', 'Backward Cache']
            for cache, title in zip(caches, titles):
                samples = cache[:16]
                x0_samples, x1_samples = zip(*samples)
                x0_samples = torch.stack(x0_samples).float() # convert to float for plotting
                x1_samples = torch.stack(x1_samples).float()
                x0_fig = plot_images(x0_samples)
                x1_fig = plot_images(x1_samples)
                logger = pl_module.logger
                logger.log_image(
                    key = title,
                    images = [x0_fig, x1_fig],
                    step = step,
                )
                plt.close('all')
                
            logger.log_metrics({
                'forward_cache_size': len(self.dsbdataset.forward_cache),
                'backward_cache_size': len(self.dsbdataset.backward_cache),
            }, step=step)
                
    def calculate_mmd(self, x : Tensor, y : Tensor) -> Tensor:
        """Compute unbiased MMD^2 between samples x and y of arbitrary shape."""
        x = x.view(x.size(0), -1)  # Flatten to (n, d)
        y = y.view(y.size(0), -1)  # (m, d)
        
        n, m = x.size(0), y.size(0)
        
        def gaussian_kernel(x : Tensor, y : Tensor, sigma : float) -> Tensor:
            x = x.view(x.size(0), -1)  # Flatten all but batch dimension: (n, d)
            y = y.view(y.size(0), -1)  # (m, d)

            x_norm = (x ** 2).sum(dim=1).unsqueeze(1)  # (n, 1)
            y_norm = (y ** 2).sum(dim=1).unsqueeze(0)  # (1, m)

            # Compute pairwise squared distances (n, m)
            dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.T)
            kernel = torch.exp(-dist / (2 * sigma ** 2))
            return kernel

        sigma = x.size(1) ** 0.5
        
        K_XX = gaussian_kernel(x, x, sigma)
        K_YY = gaussian_kernel(y, y, sigma)
        K_XY = gaussian_kernel(x, y, sigma)

        # Remove diagonal for unbiased estimate
        mmd = (
            (K_XX.sum() - torch.trace(K_XX)) / (n * (n - 1)) +
            (K_YY.sum() - torch.trace(K_YY)) / (m * (m - 1)) -
            2 * K_XY.mean()
        )

        return mmd.clamp(min=0.0)  # Ensure non-negative results