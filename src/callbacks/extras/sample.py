from .base import ExtraMetricOutput
from src.lightning_modules import DSB
from src import StepOutput, TensorDict, UnpairedAudioBatch


class GenerateSamplesExtra(ExtraMetricOutput):
    def __init__(self, num_steps: int = 1):
        self.num_steps = num_steps

    def __call__(self, pl_module: DSB, outputs: StepOutput, batch: UnpairedAudioBatch, batch_idx: int) -> TensorDict:
        sample = pl_module.sample(
            x_start=batch["x1"],
            direction="backward",
            num_steps=self.num_steps,
            encode=True,
            verbose=False,
        )
        return {
            "x0_hat": sample,
        }
