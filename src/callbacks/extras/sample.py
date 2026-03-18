from .base import ExtraMetricOutput
from src.lightning_modules import DSB
from src import StepOutput, TensorDict, UnpairedAudioBatch
from src.utils import temporary_seed


class GenerateSamplesExtra(ExtraMetricOutput):
    """
    A callback for generating samples during training, to be used as an "extra" output.
    """
    
    def __init__(self, key: str, out_key: str, **kwargs):
        super().__init__()
        self.key = key
        self.out_key = out_key
        self.kwargs = kwargs

    def __call__(self, pl_module: DSB, outputs: StepOutput, batch: UnpairedAudioBatch, batch_idx: int) -> TensorDict:
        with temporary_seed(seed=batch_idx):  # Ensure reproducibility across epochs
            sample = pl_module.sample(
                x_start=batch[self.key],
                **self.kwargs,
            )
        return {self.out_key: sample}
