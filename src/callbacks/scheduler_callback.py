import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.lightning_modules import DSB
from src import StepOutput, UnpairedAudioBatch
import torch
import matplotlib.pyplot as plt


class VisualizeSchedulerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.has_logged = False

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: DSB,
        outputs: StepOutput,
        batch: UnpairedAudioBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if self.has_logged:
            return

        self.has_logged = True

        x0, x1 = batch["x0"], batch["x1"]
        x0, x1 = pl_module.encode(x0), pl_module.encode(x1)

        # if the encodings are an image, visualize them. Only Visualize x0 (not x1) for simplicity
        if x0.dim() == 4:
            # shape (B, C, H, W)
            # the images might have varying number of channels.
            # therefore, visualize batchsize along the rows, and channels along the columns
            fig, axs = plt.subplots(
                nrows=x0.shape[0],
                ncols=x0.shape[1],
                figsize=(3.5 * x0.shape[1], 3.0 * x0.shape[0]),
                dpi=300,
                squeeze=False,  # ensure axs is always 2D array even if x0.shape[0] or x0.shape[1] is 1
            )

            for i in range(x0.shape[0]):
                for j in range(x0.shape[1]):
                    im = axs[i, j].imshow(x0[i, j].cpu(), interpolation="none")
                    axs[i, j].set_aspect("equal", adjustable="box")
                    axs[i, j].set_title(f"Sample {i}, Channel {j}")
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].grid(False)
                    fig.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)

            plt.tight_layout()

            pl_module.log_images(
                key="Scheduler/Scheduler Trajectories - Encodings",
                images=[fig],
            )

        xts = []

        for i in range(x0.shape[0]):
            n_steps = 6

            _x0, _x1 = x0[i], x1[i]

            _x0 = _x0.unsqueeze(0).repeat(n_steps, *[1] * (len(_x0.shape)))
            _x1 = _x1.unsqueeze(0).repeat(n_steps, *[1] * (len(_x1.shape)))
            timesteps = torch.linspace(0, 1, n_steps, device=_x0.device)

            scheduler_batch = pl_module.scheduler._sample_training_batch(
                x0=_x0,
                x1=_x1,
                direction="forward",  # arbitrary, since we don't use the "target" output of the scheduler
                timesteps=timesteps,
            )

            xt = scheduler_batch["xt"]
            if pl_module.scheduler.condition_on_start:
                xt, _ = xt.chunk(2, dim=1)  # remove conditioning
            xt = pl_module.decode(xt).squeeze(1).cpu()  # shape (n_steps, T)
            xts.append(xt)

        xts = torch.stack(xts, dim=0)  # shape (batch_size, n_steps, T)
        # each sample in the batch is a trajectory, i.e. a collection of n_steps waveforms showing the transition from x0 to x1
        # we would like to visualize each waveform in a grid; batch is along the rows, timesteps along the columns

        fig, axs = plt.subplots(
            nrows=xts.shape[0],
            ncols=xts.shape[1],
            figsize=(20, 2 * xts.shape[0]),
            dpi=300,
            squeeze=False,  # ensure axs is always 2D array even if xts.shape[0] or xts.shape[1] is 1
        )

        for i in range(xts.shape[0]):
            for j in range(xts.shape[1]):
                axs[i, j].plot(xts[i, j])
                axs[i, j].set_title(f"Sample {i}, Step {j}")
                axs[i, j].set_xticks([])
                axs[i, j].grid(True)

        plt.tight_layout()

        pl_module.log_images(
            key="Scheduler/Scheduler Trajectories",
            images=[fig],
        )
