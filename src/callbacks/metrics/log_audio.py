from .base import BaseMetric
from src.lightning_modules import BaseLightningModule
from src import StepOutput, TensorDict, AudioBatch
from typing import Optional, Literal


class LogAudioMetric(BaseMetric):
    """
    This "metric" doesn't compute any metrics.
    Instead, it logs specific audio files to WandB.
    """

    def __init__(
        self,
        key: str,
        where: Literal["outputs", "batch", "extras"],
        max_samples: int = 8,
        once: bool = False,
    ):
        self.key = key
        self.where = where
        assert self.where in [
            "outputs",
            "batch",
            "extras",
        ], f"Invalid value for 'where': {self.where}. Must be one of 'outputs', 'batch', or 'extras'."
        self.max_samples = max_samples
        self.once = once
        self.has_logged = False
        self.samples = []
        self.sample_rates = []

    def _is_full(self) -> bool:
        return len(self.samples) >= self.max_samples

    def add(
        self,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: AudioBatch,
        batch_idx: int,
        extras: Optional[TensorDict] = None,
    ):
        if self._is_full():
            return

        if self.where == "outputs":
            audio = outputs[self.key]
        if self.where == "batch":
            audio = batch[self.key]
        if self.where == "extras":
            audio = extras[self.key]

        batch_size = audio.shape[0]
        self.module = pl_module

        for i in range(batch_size):
            if self._is_full():
                break

            self.samples.append(audio[i].cpu().numpy().flatten())
            self.sample_rates.append(batch["sample_rate"][i])

    def compute(self) -> None:
        if not self._is_full():
            return None

        if self.has_logged and self.once:
            return None

        self.module.logger.log_audio(
            key=self.name(),
            audios=self.samples,
            step=self.module.global_step,
            sample_rate=self.sample_rates,
        )
        self.has_logged = True

    def reset(self) -> None: ...

    def name(self) -> str:
        return f"log_audio/{self.key}"
