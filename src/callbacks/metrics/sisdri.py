from .base import BaseMetric
from torch import Tensor
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from src.lightning_modules import BaseLightningModule
from src import StepOutput, Batch, TensorDict
from typing import Optional
import torch


class SISDRi(BaseMetric):
    """
    This metric calculates the scale invariant signal distortion ratio improvement (SISDRi)
    for a given clean signal, degraded signal, and model prediction.
    """

    def __init__(
        self,
        clean_key: str,
        degraded_key: str,
        output_key: str,
    ):
        self.clean_key = clean_key
        self.degraded_key = degraded_key
        self.output_key = output_key
        self.values = []

    def add(
        self,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        extras: Optional[TensorDict] = None,
    ):
        clean: Tensor = batch[self.clean_key]
        degraded: Tensor = batch[self.degraded_key]
        output = extras[self.output_key]

        clean = clean.mean(dim=1)
        degraded = degraded.mean(dim=1)
        output = output.mean(dim=1)

        sisdr_output = scale_invariant_signal_distortion_ratio(output, clean, zero_mean=True)
        sisdr_degraded = scale_invariant_signal_distortion_ratio(degraded, clean, zero_mean=True)
        improvement = (sisdr_output - sisdr_degraded).tolist()
        self.values.extend(improvement)

    def compute(self) -> TensorDict:
        values = torch.tensor(self.values)
        return {
            "mean": values.mean(),
            "std": values.std(),
        }

    def reset(self) -> None:
        self.values = []

    def name(self) -> str:
        return f"SISDRi clean='{self.clean_key}' degraded='{self.degraded_key}' prediction='{self.output_key}'"
