import torch.nn as nn
import torchaudio.transforms as T
from src.losses import BaseLossFunction
from src import Batch, ModelOutput, LossOutput
import torch


class MultiScaleMelLoss(BaseLossFunction):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.scales = [128, 256, 512, 1024, 2048]
        self.mel_bins = [20, 40, 80, 160, 210]

        self.mel_transforms = nn.ModuleList(
            [
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    win_length=n_fft,
                    hop_length=n_fft // 4,
                    n_mels=n_mels,
                    center=True,
                )
                for n_fft, n_mels in zip(self.scales, self.mel_bins)
            ]
        )

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = 0
        for mel_transform in self.mel_transforms:
            mel_output = (mel_transform(model_output["output"]) + 1e-5).log()
            mel_target = (mel_transform(batch["target"]) + 1e-5).log()

            loss += torch.nn.functional.l1_loss(mel_output, mel_target, reduction="mean")

        return LossOutput(loss=loss)
