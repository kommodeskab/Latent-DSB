from .base import BaseMetric
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from torch import Tensor
import torch
from src.lightning_modules import BaseLightningModule
from src import StepOutput, TensorDict, UnpairedAudioBatch
from typing import Optional


class DNSMOS(BaseMetric):
    """
    Mean Opinion Score (MOS) for Deep Noise Suppression (DNS) models.
    
    """
    
    
    def __init__(self, output_key: str):
        self.output_key = output_key
        self.values = []

    @torch.no_grad()
    def evaluate(self, samples: Tensor, sample_rate: int) -> Tensor:
        mos = deep_noise_suppression_mean_opinion_score(
            samples.squeeze(1), fs=sample_rate, personalized=False, device=self.device
        )
        return mos.flatten()

    def to(self, device: torch.device) -> None:
        self.device = device

    def add(
        self,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: UnpairedAudioBatch,
        batch_idx: int,
        extras: Optional[TensorDict] = None,
    ):
        samples = extras[self.output_key]
        mos = self.evaluate(samples, sample_rate=batch["sample_rate"][0])
        self.values.extend(mos.tolist())

    def compute(self) -> TensorDict:
        values = torch.tensor(self.values)
        return {
            "mean": values.mean(),
            "std": values.std(),
        }

    def reset(self) -> None:
        self.values = []

    def name(self) -> str:
        return f"DNSMOS for '{self.output_key}'"
