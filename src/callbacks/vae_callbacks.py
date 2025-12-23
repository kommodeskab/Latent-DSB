from pytorch_lightning import Callback
import pytorch_lightning as pl
from src.lightning_modules import VAEModel
from src.types import TensorDict
from torch import Tensor
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch

class VAECallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: VAEModel, outputs: TensorDict, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        if batch_idx == 0:
            logger = pl_module.logger
            
            q_z: Normal = outputs['q_z']
            p_x: Normal = outputs['p_x']
            x: Tensor = outputs['target']
            
            for i in range(min(4, batch.size(0))):
                # visualize q_z
                z_mu, z_std = q_z.mean[i], q_z.stddev[i]
                fig, axs = plt.subplots(2, 1, figsize=(6, 4))
                axs: list[plt.Axes]
                axs[0].imshow(z_mu.cpu().float().numpy(), aspect='auto', cmap='viridis')
                axs[0].set_title('Latent Mean (q_z.mu)')
                axs[1].imshow(z_std.cpu().float().numpy(), aspect='auto', cmap='viridis')
                axs[1].set_title('Latent Stddev (q_z.stddev)')
                
                for ax in axs:
                    plt.colorbar(ax.images[0], ax=ax)
                
                plt.tight_layout()
                logger.log_image(
                    key = f'Validation/Latent_Distribution/sample_{i}',
                    images = [fig],
                    step = pl_module.global_step,
                )
                plt.close(fig)
                
                # make a histogram of a sample from the latent distribution
                z_sample = q_z.rsample()[i]
                z_mu_flat = z_sample.flatten().cpu().float().numpy()
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(z_mu_flat, bins=50, density=True, alpha=0.7, color='blue')
                ax.set_title('Histogram of Latent Means (q_z.mu)')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                logger.log_image(
                    key = f'Validation/Latent_Histogram/sample_{i}',
                    images = [fig],
                    step = pl_module.global_step,
                )
                plt.close(fig)
                
                # save reconstruction as audio
                x_hat_mu = p_x.mean[i]
                x_orig = x[i]
                # make a single long audio by interleaving original and reconstructed
                x_to_log = torch.stack([x_orig, x_hat_mu], dim=1).flatten().cpu().float().numpy()
            
                logger.log_audio(
                    key = f'Validation/Reconstruction/sample_{i}',
                    audios = [x_to_log],
                    sample_rate = [16000],
                    caption = [f'Reconstructed Sample {i}'],
                    step = pl_module.global_step,
                )