from .utils import get_batch_from_dataset
from src.lightning_modules import DSB, InitDSB
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning import Callback, Trainer
from torch import Tensor
import torch
from matplotlib.figure import Figure
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from src.callbacks.metrics import MOS, KAD, WER

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
    fig, axs = plt.subplots(height, width, figsize=(width*5, height*5), dpi=300)
    axs : list[plt.Axes]
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
    while encodings.dim() < 3:
        encodings = encodings.unsqueeze(0)
    while encodings.dim() > 3:
        encodings = encodings.flatten(0, 1)
    
    n_channels = encodings.size(0)
    v_min, v_max = encodings.min(), encodings.max()
    fig, axs = plt.subplots(1, n_channels, figsize=(n_channels*5, 5))
    axs : list[plt.Axes]
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


class DiffusionCBMixin:
    def log_line_series(self, pl_module : DSB | InitDSB):
        logger = pl_module.logger
  
        def plot_line_series(y : Tensor, xlabel : str, ylabel : str) -> Figure:
            fig, _ = plt.subplots()
            plt.plot(y.flatten().numpy())
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            return fig
        
        gammas_fig = plot_line_series(pl_module.scheduler.gammas.cpu(), "timesteps", "gammas")
        gammas_bar_fig = plot_line_series(pl_module.scheduler.gammas_bar.cpu(),"timesteps", "gammas_bar")
        variance_fig = plot_line_series(pl_module.scheduler.var.cpu(), "timesteps", "variance")
        sampling_variance_fig = plot_line_series(pl_module.scheduler.sampling_var.cpu(), "timesteps", "sampling_variance")
        logger.log_image(
            key = "Scheduler",
            images = [wandb.Image(f) for f in [gammas_fig, gammas_bar_fig, variance_fig, sampling_variance_fig]],
            caption = ["gammas", "gammas_bar", "variance", "sampling_variance"],
        )
        plt.close('all')
    
    def visualize_data(self, trainer : Trainer, pl_module : DSB | InitDSB):
        logger = pl_module.logger
        device = pl_module.device
        
        def get_x0_x1(trainer : Trainer, pl_module : DSB | InitDSB, batch_size : int) -> tuple[Tensor, Tensor]:
            if isinstance(pl_module, InitDSB):
                dataset = trainer.datamodule.val_dataset
                dataset_batch = get_batch_from_dataset(dataset, batch_size=16)
                x0, x1 = dataset_batch
            else:
                x0_dataset = pl_module.x0_dataset_val
                x1_dataset = pl_module.x1_dataset_val
                x0 = get_batch_from_dataset(x0_dataset, batch_size=16)
                x1 = get_batch_from_dataset(x1_dataset, batch_size=16)
                
            return x0, x1
                
        x0, x1 = get_x0_x1(trainer, pl_module, batch_size=16)
        self.x0, self.x1 = x0, x1
            
        dim = x0.dim()
        if dim == 2:
            self.data_type = "points"
            # if we are working with points, we might aswell load some more
            x0, x1 = get_x0_x1(trainer, pl_module, batch_size=256)
        elif dim == 3:
            self.data_type = "audio"
            if isinstance(pl_module, InitDSB):
                self.sample_rate = trainer.datamodule.original_dataset.sample_rate
            else:
                self.sample_rate = pl_module.original_dataset.sample_rate
        elif dim == 4:
            self.data_type = "image"
        
        self.x0_encoded = pl_module.encode(x0.to(device))
        self.x1_encoded = pl_module.encode(x1.to(device))
        self.x0_decoded = pl_module.decode(self.x0_encoded)
        self.x1_decoded = pl_module.decode(self.x1_encoded)
        
        self.x0_encoded = self.x0_encoded.cpu()
        self.x1_encoded = self.x1_encoded.cpu()
        self.x0_decoded = self.x0_decoded.cpu()
        self.x1_decoded = self.x1_decoded.cpu()
                
        if self.data_type == "image":
            fig_x0 = plot_images(x0)
            fig_x1 = plot_images(x1)
            logger.log_image(
                key = "Initial samples",
                images = [wandb.Image(fig_x0), wandb.Image(fig_x1)],
                caption = ["x0", "x1"],
            )
            fig_x0_decoded = plot_images(self.x0_decoded)
            fig_x1_decoded = plot_images(self.x1_decoded)
            logger.log_image(
                key = "Initial decoded",
                images = [wandb.Image(fig_x0_decoded), wandb.Image(fig_x1_decoded)],
                caption = ["x0_decoded", "x1_decoded"],
            )
            self.visualize_latent_space(logger, self.x0_encoded, "x0")
            self.visualize_latent_space(logger, self.x1_encoded, "x1")
            
        elif self.data_type == "points":
            fig = plot_points([x0, x1], keys=["x0", "x1"], colors=['r', 'g'])
            logger.log_image(
                key = "Initial samples",
                images = [wandb.Image(fig)],
            )
            
        elif self.data_type == "audio":
            audios = torch.cat([x0, x1], dim=0)
            audios = [audio.flatten().numpy() for audio in audios]
            captions = ["x0"] * x0.shape[0] + ["x1"] * x1.shape[0]
                        
            logger.log_audio(
                key = "Initial samples",
                audios=audios,
                sample_rate = [self.sample_rate] * len(audios),
                caption=captions,
            )
            
            x1_decoded_audio = [audio.flatten().numpy() for audio in self.x1_decoded]
            x0_decoded_audio = [audio.flatten().numpy() for audio in self.x0_decoded]
            
            decoded_audio = x0_decoded_audio + x1_decoded_audio
            captions = ["x0"] * x0.shape[0] + ["x1"] * x1.shape[0]
            logger.log_audio(
                key = "Initial decoded (with noise)",
                audios=decoded_audio,
                sample_rate = [self.sample_rate] * len(decoded_audio),
                caption=captions,
            )
            self.visualize_latent_space(logger, self.x0_encoded, "x0")
            self.visualize_latent_space(logger, self.x1_encoded, "x1")
            
        original_size = x0.flatten(1).size(1)
        latent_size = self.x0_encoded.flatten(1).size(1)
        logger.log_metrics({
            "original_size": original_size,
            "latent_size": latent_size,
        })
        
        self.log_line_series(pl_module)
        plt.close('all')
    
    def visualize_latent_space(self, logger : WandbLogger, encodings : Tensor, title : str):
        logger.log_image(
            key=f"Encodings {title}",
            images = [wandb.Image(visualize_encodings(e)) for e in encodings.cpu()],
        )
        logger.log_image(
            key=f"Encodings histogram {title}",
            images = [wandb.Image(plot_histogram(e)) for e in encodings.cpu()],
        )
    
class FlowMatchingCB(Callback, DiffusionCBMixin):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : DSB | InitDSB):        
        self.visualize_data(trainer, pl_module)
        x0_encoded = self.x0_encoded[:5]
        x1_encoded = self.x1_encoded[:5]
        
        if self.data_type == "image":
            tensor_for_fig = torch.zeros(5, *x0_encoded.shape)
            indices = torch.linspace(0, pl_module.scheduler.timesteps.max(), 5, dtype=torch.int64)
            for i, t in enumerate(indices):
                if i == 0:
                    tensor_for_fig[i] = x0_encoded
                    continue
                xt, _ = pl_module.scheduler.forward_process(x0_encoded, x1_encoded, t)
                tensor_for_fig[i] = xt
            
            tensor_for_fig = tensor_for_fig.flatten(0, 1).to(pl_module.device)
            decoded = pl_module.decode(tensor_for_fig).cpu()
            fig = plot_images(decoded)
            pl_module.logger.log_image(
                key = "Initial trajectory",
                images = [wandb.Image(fig)],
            )
        
        elif self.data_type == "audio":
            self.mos = MOS(pl_module.device)
            # self.kad = KAD()            
            
        if isinstance(pl_module, DSB):
            # it is a good idea to check the quality of the initial samples
            # therefore, we momentarily set the model to "training forward" in order to sample from the forward model
            # samples can be generated via self.on_validation_end
            # we only test the forward process at the very start of training
            if pl_module.training_backward and pl_module.DSB_iteration == 1:
                pl_module.DSB_iteration = 0
                print("Testing initial backward process..")
                self.on_validation_end(trainer, pl_module)
                pl_module.training_backward = False
                print("Testing initial forward process..")
                self.on_validation_end(trainer, pl_module)
                pl_module.training_backward = True
                pl_module.DSB_iteration = 1
        
        plt.close('all')
        
    def on_validation_end(self, trainer : Trainer, pl_module : DSB | InitDSB):
        # we fix the validation seed to fairly compare samples and metrics across epochs
        with pl_module.fix_validation_seed():
            logger = pl_module.logger
            device = pl_module.device
            
            if isinstance(pl_module, InitDSB):
                x1_encoded = self.x1_encoded.to(device)
                trajectory = pl_module.sample(x1_encoded, return_trajectory=True, show_progress=True)
                caption = "x0 Samples"
            else:
                backward = pl_module.training_backward
                x_start = self.x1_encoded if backward else self.x0_encoded
                x_start = x_start.to(device)
                trajectory = pl_module.sample(x_start, forward=not backward, return_trajectory=True, show_progress=True)
                iteration = pl_module.DSB_iteration
                backward_str = "x0 samples" if backward else "x1 samples"
                caption = f"iteration_{iteration}/{backward_str}"
            
            final_samples = trajectory[-1]
            initial_samples = trajectory[0]
            # some encoders have randomness in them, fx VAE's
            final_decoded = pl_module.decode(final_samples).cpu()
            initial_decoded = pl_module.decode(initial_samples).cpu()
            
            if self.data_type == 'image':
                fig = plot_images(final_decoded)
                logger.log_image(
                    key = caption,
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
                fig = plot_points([trajectory.cpu(), final_decoded, initial_decoded], keys=["trajectory", "x0_hat", "x1"], colors=['b', 'r', 'g'])
                logger.log_image(
                    key = caption,
                    images = [wandb.Image(fig)],
                    step = pl_module.global_step,
                )
                
            elif self.data_type == "audio":
                audios = [audio.flatten().numpy() for audio in final_decoded]
                logger.log_audio(
                    key = caption,
                    audios=audios,
                    sample_rate = [self.sample_rate] * len(audios),
                    step = pl_module.global_step,
                )
                
                mos = self.mos.evaluate(final_decoded, self.sample_rate)
            
                if isinstance(pl_module, InitDSB):
                    mos_caption = "x0 MOS"
                    
                    logger.log_metrics({
                        mos_caption: mos,
                    }, step=pl_module.global_step)
                    
                else:
                    mos_caption = f"iteration_{iteration}/{backward_str} MOS"
                    
                    logger.log_metrics({
                        mos_caption: mos,
                        'curr_num_iters': pl_module.curr_num_iters,
                    }, step=pl_module.global_step)
            
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
        self.visualize_data(trainer, pl_module)
        x0 = self.x0_encoded.to(device)
        trajectory = pl_module.sample(x0, forward=True, return_trajectory=True, show_progress=True, noise='inference')
        sampled_trajectory = self.sample_from_trajectory(trajectory, 5, 5)
        sampled_trajectory = pl_module.decode(sampled_trajectory)
        # sampled.trajectory shape: (25, ...)
        
        if self.data_type == "image":
            sampled_trajectory_fig = plot_images(sampled_trajectory.cpu())
            pl_module.logger.log_image(
                key = "Initial trajectory",
                images = [wandb.Image(sampled_trajectory_fig)],
            )
        
        elif self.data_type == "audio":
            shape = sampled_trajectory.size()
            sampled_trajectory = sampled_trajectory.view(5, 5, *shape[1:])
            for i in range(5):
                sample = sampled_trajectory[:, i] # shape (5, channels, time)
                audios = [sample.flatten().numpy()]
                pl_module.logger.log_audio(
                    key = f"Initial trajectory/{i}",
                    audios = audios,
                    sample_rate = [self.sample_rate],
                )
            
            self.mos = MOS(pl_module.device)
            
        plt.close('all')
        
    def on_validation_end(self, trainer : Trainer, pl_module : DSB):
        device      = pl_module.device
        logger      = pl_module.logger
        is_backward = pl_module.training_backward
        iteration   = pl_module.DSB_iteration
        
        x_start = self.x1_encoded if is_backward else self.x0_encoded
        x_start = x_start.to(device)
        
        # samples shape (num_steps, 16, c, h, w)
        samples = pl_module.sample(
            x_start = x_start,
            forward = not is_backward,
            return_trajectory = True,
        )
        final_samples = samples[-1]
        final_samples = pl_module.decode(final_samples) # shape: (16, ...)
        sampled_trajectory = self.sample_from_trajectory(samples, 5, 5)
        sampled_trajectory = pl_module.decode(sampled_trajectory)
        
        if self.data_type == "image":
            sampled_trajectory_fig = plot_images(sampled_trajectory.cpu())
            samples_fig = plot_images(final_samples.cpu())

            caption = "forward" if not is_backward else "backward"
            logger.log_image(
                key = f"iteration_{iteration}/{caption}_samples",
                images = [wandb.Image(samples_fig)],
                step = pl_module.global_step,
            )
            
            logger.log_image(
                key = f"iteration_{iteration}/{caption}_trajectory",
                images = [wandb.Image(sampled_trajectory_fig)],
                step = pl_module.global_step,
            )
            
        elif self.data_type == "audio":
            audios = [audio.flatten().numpy() for audio in final_samples]
            logger.log_audio(
                key = f"iteration_{iteration}/samples",
                audios = audios,
                sample_rate = [self.sample_rate] * len(audios),
                step = pl_module.global_step,
            )
            
            mos = self.mos.evaluate(final_samples)
            logger.log_metrics({
                "MOS": mos,
            }, step=pl_module.global_step)
            
        plt.close('all')