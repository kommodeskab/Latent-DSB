from pytorch_lightning import Callback, Trainer
from src.lightning_modules.dsb import DSB
from src.data_modules import FlowMatchingDM
import torch
from src.callbacks.utils import get_batch_from_dataset
from src.callbacks.callbacks import plot_images
import matplotlib.pyplot as plt
from typing import Literal

class DSBCallback(Callback):
    def __init__(
        self,
        num_steps : int = 50,
        scheduler_type : str = 'cosine',
        what_media : Literal['image', 'audio'] = 'image',
    ):
        super().__init__()
        self.num_steps = num_steps
        self.scheduler_type = scheduler_type
        self.what_media = what_media
        
    def on_train_start(self, trainer : Trainer, pl_module : DSB):
        self.datamodule : FlowMatchingDM = trainer.datamodule
        self.num_workers = self.datamodule.num_workers
        
        logger = pl_module.logger
        num_samples = 16
        
        self.x0_batch = get_batch_from_dataset(self.datamodule.x0_valset, num_samples, shuffle=False)
        self.x1_batch = get_batch_from_dataset(self.datamodule.x1_valset, num_samples, shuffle=False)

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
            x0_fig = plot_images(self.x0_batch)
            x1_fig = plot_images(self.x1_batch)
            logger.log_image(
                key = 'Initial Samples',
                images = [x0_fig, x1_fig],
                step = pl_module.global_step,
            )
            
            plt.close('all')
            
        elif self.what_media == 'audio':
            self.sample_rate : int = self.datamodule.original_dataset.sample_rate
            x0_audio = [audio.cpu().flatten().numpy() for audio in self.x0_batch]
            x1_audio = [audio.cpu().flatten().numpy() for audio in self.x1_batch]
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
            x0_fig = plot_images(forward_samples.cpu())
            x1_fig = plot_images(backward_samples.cpu())
            
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
            x0_audio = [audio.cpu().flatten().numpy() for audio in backward_samples]
            x1_audio = [audio.cpu().flatten().numpy() for audio in forward_samples]
            audios = x0_audio + x1_audio
            caption = ["x0"] * len(x0_audio) + ["x1"] * len(x1_audio)
            
            logger.log_audio(
                key = 'Samples',
                audios = audios,
                sample_rate = [self.sample_rate] * len(audios),
                caption = caption,
                step = step,
            )
        
    def on_validation_end(self, trainer : Trainer, pl_module : DSB):
        self.visualize_samples(pl_module)
        