import pytorch_lightning as pl
from .utils import get_batches, get_batch_from_dataset
from src.lightning_modules import DSB, FM, VAE
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning import Callback, Trainer
from torch import Tensor
import torch
from matplotlib.figure import Figure

def plot_samples(samples : Tensor) -> Figure:
    # assume samples have shape (16, c, h, w)
    # and have values between -1 and 1
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    cmap = 'gray' if samples.shape[1] == 1 else None
    samples = samples.permute(0, 2, 3, 1)
    fig, axs = plt.subplots(4, 4, figsize = (10, 10))
    for i in range(16):
        ax = axs[i // 4, i % 4]
        ax.imshow(samples[i], cmap = cmap)
        ax.axis('off')
    return fig

class VAECB(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : VAE):
        logger = pl_module.logger
        device = pl_module.device
        
        dataset = trainer.datamodule.val_dataset
        self.x0 = get_batch_from_dataset(dataset, batch_size=16)
        fig = plot_samples(self.x0)
        logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(fig)],
        )
        plt.close(fig)
        
    def on_validation_end(self, trainer : Trainer, pl_module : VAE):
        logger = pl_module.logger
        device = pl_module.device
        
        x0 = self.x0.to(device)
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

class TestEncoderDecoderCB(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : DSB):
        logger = pl_module.logger
        device = pl_module.device
        
        x0, x1 = get_batches(trainer, batch_size = 1, shuffle = False)
        x0 = x0.to(device)
        x1 = x1.to(device)
        
        x0_encoded = pl_module.encode(x0)
        x0_decoded = pl_module.decode(x0_encoded)
        
        x1_encoded = pl_module.encode(x1)
        x1_decoded = pl_module.decode(x1_encoded)
        
        x0, x1 = x0.cpu(), x1.cpu()
        x0_decoded, x1_decoded = x0_decoded.cpu(), x1_decoded.cpu()
        
        # shape of x0 and x1: (1, c, h, w)
        cmap = 'gray' if x0.shape[1] == 1 else None
        
        fig, axs = plt.subplots(2, 2, figsize = (10, 10))
        axs[0, 0].imshow(x0[0, 0], cmap = cmap)
        axs[0, 0].set_title('x0')
        
        axs[0, 1].imshow(x0_decoded[0, 0], cmap = cmap)
        axs[0, 1].set_title('x0 decoded')
        
        axs[1, 0].imshow(x1[0, 0], cmap = cmap)
        axs[1, 0].set_title('x1')
        
        axs[1, 1].imshow(x1_decoded[0, 0], cmap = cmap)
        axs[1, 1].set_title('x1 decoded')
        
        for ax in axs.flatten():
            ax.axis('off')
        
        logger.log_image(
            key="test_encoder_decoder",
            images=[wandb.Image(fig)],
            step=trainer.global_step
        )
        plt.close(fig)
        
        # get the total size (number of pixels) of the images and the latent space
        original_size = x0.flatten(1).size(1)
        latent_size = x0_encoded.flatten(1).size(1)
        # log the sizes
        logger.log_metrics({
                "original_size": original_size,
                "latent_size": latent_size,
            })
        
class PlotSamplesCB(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : DSB):
        device = pl_module.device
        logger = pl_module.logger
        
        x0, x1 = get_batches(trainer, batch_size=16, shuffle=False)
        
        # move to device
        x0 = x0.to(device)
        x1 = x1.to(device)
        
        # encode the samples
        x0_encoded = pl_module.encode(x0)
        x1_encoded = pl_module.encode(x1)
        
        self.x0, self.x1 = x0.cpu(), x1.cpu()
        self.x0_encoded, self.x1_encoded = x0_encoded.cpu(), x1_encoded.cpu()
        
        # plot the initial samples
        x0_fig = plot_samples(self.x0)
        x1_fig = plot_samples(self.x1)
        
        logger.log_image(
            key = "Initial samples",
            images = [wandb.Image(x0_fig), wandb.Image(x1_fig)],
            caption = ["x0", "x1"],
        )
        plt.close(x0_fig)
        plt.close(x1_fig)
        
    def on_validation_end(self, trainer : Trainer, pl_module : DSB):
        device      = pl_module.device
        logger      = pl_module.logger
        is_backward = pl_module.training_backward
        iteration   = pl_module.DSB_iteration
        
        x_start = self.x1_encoded if is_backward else self.x0_encoded
        x_start = x_start.to(device)
        
        samples = pl_module.sample(
            x_start = x_start,
            forward = not is_backward,
            return_trajectory = False,
            clamp = True,
        )
        samples_decoded = pl_module.decode(samples) # shape: (16, c, h, w)
        samples_decoded = samples_decoded.cpu() # move to cpu for plotting
        samples_fig = plot_samples(samples_decoded)

        caption = "Forward" if not is_backward else "Backward"
        logger.log_image(
            key = f"Samples iteration {iteration}",
            images = [wandb.Image(samples_fig)],
            caption = [caption],
            step = pl_module.global_step,
        )
        plt.close(samples_fig)