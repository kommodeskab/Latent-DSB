import pytorch_lightning as pl
from .utils import get_batches
from src.lightning_modules.schrodinger_bridge import DSB
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning import Callback, Trainer
from torch import Tensor
import torch
from matplotlib.figure import Figure

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
        
    def plot_samples(self, samples : Tensor) -> Figure:
        # assume samples have shape (16, c, h, w)
        # and have values between -1 and 1
        samples = (samples + 1) / 2
        cmap = 'gray' if samples.shape[1] == 1 else None
        fig, axs = plt.subplots(4, 4, figsize = (10, 10))
        for i in range(16):
            ax = axs[i // 4, i % 4]
            ax.imshow(samples[i, 0], cmap = cmap)
            ax.axis('off')
        return fig
        
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
        x0_fig = self.plot_samples(self.x0)
        x1_fig = self.plot_samples(self.x1)
        
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
        samples_fig = self.plot_samples(samples_decoded)

        caption = "Forward" if not is_backward else "Backward"
        logger.log_image(
            key = f"Samples iteration {iteration}",
            images = [wandb.Image(samples_fig)],
            caption = [caption],
            step = pl_module.global_step,
        )
        plt.close(samples_fig)