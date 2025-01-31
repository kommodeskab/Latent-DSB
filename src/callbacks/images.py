from .utils import get_batches, get_batch_from_dataset
from src.lightning_modules import DSB, FM, VAE
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning import Callback, Trainer
from torch import Tensor
import torch
from matplotlib.figure import Figure
import time

def plot_samples(samples : Tensor) -> Figure:
    # assume samples have shape (k, c, h, w)
    # and have values between -1 and 1
    k = int(samples.size(0) ** 0.5)
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    cmap = 'gray' if samples.shape[1] == 1 else None
    samples = samples.permute(0, 2, 3, 1)
    fig, axs = plt.subplots(k, k, figsize = (10, 10))
    for i in range(k * k):
        ax = axs[i // k, i % k]
        ax.imshow(samples[i], cmap = cmap)
        ax.axis('off')
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
        fig = plot_samples(self.x0.cpu())
        logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(fig)],
        )
        plt.close(fig)
        
        encoded = pl_module.encode(self.x0)
        encoded_size = encoded.flatten(1).size(1)
        original_size = self.x0.flatten(1).size(1)
        logger.log_metrics({
            "encoded_size": encoded_size,
            "original_size": original_size,
        })
        
    def on_validation_end(self, trainer : Trainer, pl_module : VAE):
        logger = pl_module.logger
        device = pl_module.device
        
        x0 = self.x0
        reconstructed = pl_module.encode_decode(x0)
        fig = plot_samples(reconstructed.cpu())
        logger.log_image(
            key = "Reconstructed samples",
            images = [wandb.Image(fig)],
            step = pl_module.global_step,
        )
        plt.close(fig)

class FlowMatchingCB(Callback):
    def __init__(
        self,
        test_initial_samples : bool = False,
        ):
        super().__init__()
        self.test_initial_samples = test_initial_samples
        
    def on_train_start(self, trainer : Trainer, pl_module : FM):
        logger = pl_module.logger
        device = pl_module.device
        
        dataset = trainer.datamodule.val_dataset
        x0 = get_batch_from_dataset(dataset, batch_size=16) 
        
        fig = plot_samples(x0)
        logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(fig)],
        )
        plt.close(fig)
        
        encoded = pl_module.encode(x0.to(device))
        self.noise = torch.randn(*encoded.shape)
        decoded = pl_module.decode(encoded)
        
        fig = plot_samples(decoded.cpu())
        logger.log_image(
            key = "Initial decoded",
            images = [wandb.Image(fig)],
        )
        plt.close(fig)
        
        original_size = x0.flatten(1).size(1)
        latent_size = encoded.flatten(1).size(1)
        logger.log_metrics({
            "original_size": original_size,
            "latent_size": latent_size,
        })
        
        if self.test_initial_samples:
            self.on_validation_end(trainer, pl_module)
        
    def on_validation_end(self, trainer : Trainer, pl_module : FM):
        logger = pl_module.logger
        device = pl_module.device
        
        noise = self.noise.to(device)
        samples = pl_module.sample(noise)
        samples = pl_module.decode(samples).cpu()
        fig = plot_samples(samples)
        logger.log_image(
            key = "Samples",
            images = [wandb.Image(fig)],
            step = pl_module.global_step,
        )
        plt.close(fig)


class DSBCB(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : DSB):
        device = pl_module.device
        logger = pl_module.logger
        
        x0, x1 = get_batches(trainer, batch_size=16, shuffle=False)
                
        # plot the initial samples
        x0_fig = plot_samples(x0.cpu())
        x1_fig = plot_samples(x1.cpu())
        
        logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(x0_fig), wandb.Image(x1_fig)],
            caption = ["x0", "x1"],
        )
        plt.close(x0_fig)
        plt.close(x1_fig)
        
        # move to device
        x0 = x0.to(device)
        x1 = x1.to(device)
        
        # encode the samples
        x0_encoded = pl_module.encode(x0)
        x1_encoded = pl_module.encode(x1)
        self.x0_encoded, self.x1_encoded = x0_encoded, x1_encoded
        
        x0_decoded = pl_module.decode(x0_encoded)
        x1_decoded = pl_module.decode(x1_encoded)
        
        fig_1 = plot_samples(x0_decoded.cpu())
        fig_2 = plot_samples(x1_decoded.cpu())
        
        logger.log_image(
            key = "Initial decoded",
            images = [wandb.Image(fig_1), wandb.Image(fig_2)],
            caption = ["x0", "x1"],
        )
        plt.close(fig_1)
        plt.close(fig_2)
        
        original_size = x0.flatten(1).size(1)
        encoded_size = x0_encoded.flatten(1).size(1)
        
        t1 = time.time()
        initial_forward_process = pl_module.sample(x0_encoded, forward=True, return_trajectory=True)
        time_to_sample = time.time() - t1
        logger.log_metrics({
            "original_size": original_size,
            "encoded_size": encoded_size,
            "time_to_sample": time_to_sample,
        })
        
        selected_indices = torch.linspace(0, len(initial_forward_process) - 1, 5).round().long().to(device)
        selected_forward_process = initial_forward_process[selected_indices, :5] # shape (5, 5, c, h, w)
        selected_forward_process = selected_forward_process.flatten(0, 1) # shape (25, c, h, w)
        selected_forward_process = pl_module.decode(selected_forward_process)
        selected_forward_process_fig = plot_samples(selected_forward_process.cpu())
        
        logger.log_image(
            key = "Initial forward process",
            images = [wandb.Image(selected_forward_process_fig)],
            step = pl_module.global_step,
        )
        plt.close(selected_forward_process_fig)
        
        fig_1 = plot_graph(pl_module.gamma_scheduler.gammas, title="Gammas")
        fig_2 = plot_graph(pl_module.gamma_scheduler.gammas_bar, title="Gammas bar")
        fig_3 = plot_graph(pl_module.gamma_scheduler.sigma_backward, title="Sigma backward")
        fig_4 = plot_graph(pl_module.gamma_scheduler.sigma_forward, title="Sigma forward")
        
        logger.log_image(
            key = "Gammas",
            images = [wandb.Image(fig_1), wandb.Image(fig_2), wandb.Image(fig_3), wandb.Image(fig_4)],
            caption = ["Gammas", "Gammas bar", "Sigma backward", "Sigma forward"],
        )
        plt.close(fig_1)
        plt.close(fig_2)
        plt.close(fig_3)
        plt.close(fig_4)
        
        
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
        final_samples = samples[0] if is_backward else samples[-1]
        final_samples = pl_module.decode(final_samples) # shape: (16, c, h, w)
        samples_fig = plot_samples(final_samples.cpu())

        caption = "forward" if not is_backward else "backward"
        logger.log_image(
            key = f"iteration_{iteration}/{caption}_samples",
            images = [wandb.Image(samples_fig)],
            step = pl_module.global_step,
        )
        plt.close(samples_fig)
        
        selected_indices = torch.linspace(0, len(samples) - 1, 5).round().long().to(device)
        selected_samples = samples[selected_indices, :5]
        selected_samples = selected_samples.flatten(0, 1)
        selected_samples = pl_module.decode(selected_samples)
        selected_samples_fig = plot_samples(selected_samples.cpu())
        logger.log_image(
            key = f"iteration_{iteration}/{caption}_trajectory",
            images = [wandb.Image(selected_samples_fig)],
            step = pl_module.global_step,
        )
        plt.close(selected_samples_fig)
        