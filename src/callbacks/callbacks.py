from .utils import get_batches, get_batch_from_dataset
from src.lightning_modules import DSB, FM, VAE, Classifier
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning import Callback, Trainer
from torch import Tensor
import torch
from matplotlib.figure import Figure
from torch.utils.data import Dataset, Subset
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from typing import Literal

def plot_images(samples : Tensor, height : int | None = None, width : int | None = None) -> Figure:
    # assume samples have shape (k, c, h, w)
    # and have values between -1 and 1
    if height is None and width is None:
        k = int(samples.size(0) ** 0.5)
        height = width = k

    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    cmap = 'gray' if samples.shape[1] == 1 else None
    samples = samples.permute(0, 2, 3, 1)
    fig, axs = plt.subplots(height, width, figsize=(width*5, height*5))
    for i in range(height):
        for j in range(width):
            ax = axs[i, j] if height > 1 else axs[j]
            ax.imshow(samples[i * height + j], cmap=cmap)
            ax.axis('off')
    return fig

def plot_points(points : list[Tensor], keys : list[str], colors : list[str]) -> Figure:
    fig, ax = plt.subplots()
    for point, color, key in zip(points, colors, keys):
        dim = point.dim()
        if dim == 2:
            # points have shape (n, 2)
            ax.scatter(point[:, 0], point[:, 1], c=color, label=key, s=10)
        elif dim == 3:
            # points have shape (num_steps, n, 2)
            # visualize the trajectory for each point
            # only visualize some of the trajectory
            n_points = 7
            indices = torch.linspace(0, point.size(0) - 1, n_points).round().to(torch.int64)
            point = point[indices]

            batch_size = point.size(1)
            for i in range(batch_size):
                label = key if i == 0 else None
                # only visualize some of the points
                ax.scatter(point[:, i, 0], point[:, i, 1], c=color, label=label, s=1, alpha=0.3)
                
    ax.legend()
    return fig

def visualize_encodings(encodings : Tensor) -> tuple[Figure, Figure]:
    assert encodings.dim() == 3, "Encodings must have shape (c, h, w)"
    n_channels = encodings.size(0)
    v_min, v_max = encodings.min().item(), encodings.max().item()
    fig, axs = plt.subplots(1, n_channels, figsize=(n_channels*5, 5))
    axs = np.atleast_1d(axs)
    norm = plt.Normalize(vmin=v_min, vmax=v_max)
    cmap = 'gray' if n_channels == 1 else 'viridis'
    for i in range(n_channels):
        im = axs[i].imshow(encodings[i], cmap=cmap, norm=norm)
        axs[i].axis('off')
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Intensity')
    
    return fig

def plot_histogram(data : Tensor) -> Figure:   
    fig, ax = plt.subplots()
    ax.hist(data.flatten(), bins=100, density=True)
    return fig

def plot_graph(y, x = None, title = None, xlabel = None, ylabel = None):
    fig, _ = plt.subplots()
    if x is None:
        x = range(len(y))
    plt.plot(x, y)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    return fig

class VAECB(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : VAE):
        logger = pl_module.logger
        device = pl_module.device
        
        dataset = trainer.datamodule.val_dataset
        self.x0 = get_batch_from_dataset(dataset, batch_size=16).to(device)
        fig = plot_images(self.x0.cpu())
        logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(fig)],
        )
        
        encoded = pl_module.encode(self.x0)
        encoded_size = encoded.flatten(1).size(1)
        original_size = self.x0.flatten(1).size(1)
        logger.log_metrics({
            "encoded_size": encoded_size,
            "original_size": original_size,
        })
        plt.close('all')
        
    def on_validation_end(self, trainer : Trainer, pl_module : VAE):
        logger = pl_module.logger
        
        x0 = self.x0
        reconstructed = pl_module.encode_decode(x0)
        fig = plot_images(reconstructed.cpu())
        logger.log_image(
            key = "Reconstructed samples",
            images = [wandb.Image(fig)],
            step = pl_module.global_step,
        )
        plt.close('all')
        
class DiffusionCBMixin:
    def log_line_series(self, pl_module : FM | DSB):
        logger = pl_module.logger
        gammas = pl_module.scheduler.gammas.tolist()
        gammas_bar = pl_module.scheduler.gammas_bar.tolist()
        
        logger.log_metrics({
            "gammas": wandb.plot.line_series(
                xs = torch.linspace(0, 1, len(gammas)).tolist(),
                ys = [gammas],
                keys = ["gamma"],
            ),
            "gammas_bar": wandb.plot.line_series(
                xs = torch.linspace(0, 1, len(gammas_bar)).tolist(),
                ys = [gammas_bar],
                keys = ["gamma_bar"],
            ),
        })
    
    def visualize_latent_space(self, logger : WandbLogger, encodings : Tensor):
        logger.log_image(
            key="Encodings",
            images = [wandb.Image(visualize_encodings(e)) for e in encodings.cpu()],
        )
        logger.log_image(
            key="Encodings histogram",
            images = [wandb.Image(plot_histogram(e)) for e in encodings.cpu()],
        )
    
class FlowMatchingCB(Callback, DiffusionCBMixin):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : FM):
        logger = pl_module.logger
        device = pl_module.device
        
        dataset = trainer.datamodule.val_dataset
        dataset_batch = get_batch_from_dataset(dataset, batch_size=16)
        x0, x1 = dataset_batch
        dim = x0.dim()
        if dim == 2:
            self.data_type = "points"
            # if we are working with points, we might aswell load some more
            dataset_batch = get_batch_from_dataset(dataset, batch_size=256)
            x0, x1 = dataset_batch
        elif dim == 3:
            self.data_type = "audio"
            self.sample_rate = pl_module.encoder_decoder.sample_rate
        elif dim == 4:
            self.data_type = "image"
            
        self.x1_encoded = pl_module.encode(x1.to(device))
        encoded = pl_module.encode(x0.to(device), add_noise=True)
        decoded = pl_module.decode(encoded).cpu()
        
        if self.data_type == "image":
            fig_x0 = plot_images(x0)
            fig_x1 = plot_images(x1)
            logger.log_image(
                key = "Initial samples",
                images = [wandb.Image(fig_x0), wandb.Image(fig_x1)],
                caption = ["x0", "x1"],
            )
            fig = plot_images(decoded)
            logger.log_image(
                key = "Initial decoded (with noise)",
                images = [wandb.Image(fig)],
            )
            self.visualize_latent_space(logger, encoded)
            
        elif self.data_type == "points":
            fig = plot_points([x0, x1], keys=["x0", "x1"], colors=['r', 'g'])
            logger.log_image(
                key = "Initial samples",
                images = [wandb.Image(fig)],
            )
            
        elif self.data_type == "audio":
            logger.log_audio(
                key = "Initial samples",
                audios=x0.tolist() + x1.tolist(),
                sample_rate = self.sample_rate,
            )
            self.visualize_latent_space(logger, encoded)
            
        original_size = x0.flatten(1).size(1)
        latent_size = encoded.flatten(1).size(1)
        logger.log_metrics({
            "original_size": original_size,
            "latent_size": latent_size,
        })
        
        self.log_line_series(pl_module)
        plt.close('all')
        
    def on_validation_end(self, trainer : Trainer, pl_module : FM):
        logger = pl_module.logger
        
        x1_encoded = self.x1_encoded        
        trajectory = pl_module.sample(x1_encoded, return_trajectory=True)
        
        final_samples = trajectory[-1]
        initial_samples = trajectory[0]
        x0_hat = pl_module.decode(final_samples).cpu()
        x1 = pl_module.decode(initial_samples).cpu()
        
        if self.data_type == 'image':
            fig = plot_images(x0_hat)
            logger.log_image(
                key = "Samples",
                images = [wandb.Image(fig)],
                step = pl_module.global_step,
            )
            selected_idxs = torch.linspace(0, trajectory.size(0) - 1, 5).round().to(torch.int64)
            trajectory = trajectory[selected_idxs, :5].flatten(0, 1)
            trajectory = pl_module.decode(trajectory).cpu()
            fig = plot_images(trajectory)
            logger.log_image(
                key = "Trajectory",
                images = [wandb.Image(fig)],
                step = pl_module.global_step,
            )
        
        elif self.data_type == "points":
            fig = plot_points([trajectory.cpu(), x0_hat, x1], keys=["trajectory", "x0_hat", "x1"], colors=['b', 'r', 'g'])
            logger.log_image(
                key = "Samples",
                images = [wandb.Image(fig)],
                step = pl_module.global_step,
            )
            
        elif self.data_type == "audio":
            logger.log_audio(
                key = "Samples",
                audios=x0_hat.tolist(),
                sample_rate = self.sample_rate,
                step = pl_module.global_step,
            )
        
        plt.close('all')

class TestFMOnDatasetCB(Callback):
    def __init__(self, dataset : Dataset):
        super().__init__()
        self.dataset = dataset
    
    def on_train_start(self, trainer : Trainer, pl_module : FM):
        device = pl_module.device
        logger = pl_module.logger
        
        dataset = self.dataset
        x0 = get_batch_from_dataset(dataset, batch_size=16).to(device)
        self.x0_encoded = pl_module.encode(x0, add_noise=True)
        x0_decoded = pl_module.decode(self.x0_encoded)
        fig = plot_images(x0_decoded.cpu())
        logger.log_image(
            key = "Initial decoded for test dataset",
            images = [wandb.Image(fig)],
        )
        plt.close('all')
        
    def on_validation_end(self, trainer : Trainer, pl_module : FM):
        logger = pl_module.logger
        
        x0_encoded = self.x0_encoded
        samples = pl_module.sample(x0_encoded)
        samples = pl_module.decode(samples).cpu()
        fig = plot_images(samples)
        logger.log_image(
            key = "Samples for test dataset",
            images = [wandb.Image(fig)],
            step = pl_module.global_step,
        )
        plt.close('all')
        
class DSBCB(Callback, DiffusionCBMixin):
    def __init__(self):
        super().__init__()
        
    def sample_from_trajectory(self, trajectory : Tensor, n_timesteps: int, n_samples : int) -> Tensor:
        # trajectory shape: (num_steps, batch_size, ...)
        # get sample of shape (num_samples * batch_size, ...)
        num_steps = trajectory.size(0)
        indices = torch.linspace(0, num_steps - 1, n_timesteps).round().to(torch.int64)
        samples = trajectory[indices, :n_samples]
        samples = samples.flatten(0, 1)
        return samples
        
    def on_train_start(self, trainer : Trainer, pl_module : DSB):
        device = pl_module.device
        logger = pl_module.logger
        
        x0, x1 = get_batches(trainer, batch_size=16, shuffle=False)
                
        # plot the initial samples
        x0_fig = plot_images(x0.cpu())
        x1_fig = plot_images(x1.cpu())
        
        logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(x0_fig), wandb.Image(x1_fig)],
            caption = ["x0", "x1"],
        )
        
        # move to device
        x0 = x0.to(device)
        x1 = x1.to(device)
        
        # encode the samples
        x0_encoded = pl_module.encode(x0)
        x1_encoded = pl_module.encode(x1)
        self.x0_encoded, self.x1_encoded = x0_encoded, x1_encoded
        
        x0_decoded = pl_module.decode(x0_encoded)
        x1_decoded = pl_module.decode(x1_encoded)
        
        fig_1 = plot_images(x0_decoded.cpu())
        fig_2 = plot_images(x1_decoded.cpu())
        
        logger.log_image(
            key = "Initial decoded",
            images = [wandb.Image(fig_1), wandb.Image(fig_2)],
            caption = ["x0", "x1"],
        )
        
        original_size = x0.flatten(1).size(1)
        encoded_size = x0_encoded.flatten(1).size(1)
        
        initial_forward_process = pl_module.sample(x0_encoded, forward=True, return_trajectory=True, use_initial_forward_sampling=True)
        logger.log_metrics({
            "original_size": original_size,
            "encoded_size": encoded_size,
        })
        
        sampled_trajectory = self.sample_from_trajectory(initial_forward_process, 5, 5)
        sampled_trajectory = pl_module.decode(sampled_trajectory)
        sampled_trajectory_fig = plot_images(sampled_trajectory.cpu())
        
        logger.log_image(
            key = "Initial forward process",
            images = [wandb.Image(sampled_trajectory_fig)],
            step = pl_module.global_step,
        )
        
        self.visualize_latent_space(logger, x0_encoded)
        self.log_line_series(pl_module)
        plt.close('all')
        
    def on_validation_end(self, trainer : Trainer, pl_module : DSB):
        device      = pl_module.device
        logger      = pl_module.logger
        is_backward = pl_module.training_backward
        iteration   = pl_module.DSB_iteration
        
        x_start = self.x1_encoded if is_backward else self.x0_encoded
        
        # samples shape (num_steps, 16, c, h, w)
        samples = pl_module.sample(
            x_start = x_start,
            forward = not is_backward,
            return_trajectory = True,
        )
        final_samples = samples[-1]
        final_samples = pl_module.decode(final_samples) # shape: (16, c, h, w)
        samples_fig = plot_images(final_samples.cpu())

        caption = "forward" if not is_backward else "backward"
        logger.log_image(
            key = f"iteration_{iteration}/{caption}_samples",
            images = [wandb.Image(samples_fig)],
            step = pl_module.global_step,
        )
        
        sampled_trajectory = self.sample_from_trajectory(samples, 5, 5)
        sampled_trajectory = pl_module.decode(sampled_trajectory)
        sampled_trajectory_fig = plot_images(sampled_trajectory.cpu())
        logger.log_image(
            key = f"iteration_{iteration}/{caption}_trajectory",
            images = [wandb.Image(sampled_trajectory_fig)],
            step = pl_module.global_step,
        )
        plt.close('all')
        
        
class ClassificationCB(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : Classifier):
        dataset = trainer.datamodule.val_dataset
        batch = get_batch_from_dataset(dataset, batch_size=32)
        x0, x1 = batch
        x0, x1 = x0.to(pl_module.device), x1.to(pl_module.device)
        self.x0_encoded = pl_module.encoder_decoder.encode(x0)
        self.x1_encoded = pl_module.encoder_decoder.encode(x1)
        
        x0_decoded = pl_module.encoder_decoder.decode(self.x0_encoded)
        x1_decoded = pl_module.encoder_decoder.decode(self.x1_encoded)
        
        fig_x0 = plot_images(x0_decoded.cpu()[:16])
        fig_x1 = plot_images(x1_decoded.cpu()[:16])
        pl_module.logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(fig_x0), wandb.Image(fig_x1)],
            caption = ["x0", "x1"],
        )
        plt.close('all')