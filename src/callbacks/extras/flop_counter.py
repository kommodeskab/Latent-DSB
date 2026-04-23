from torch.utils.flop_counter import FlopCounterMode
from torch import tensor
from .base import ExtraMetricOutput
from src.lightning_modules import DSB
from src import StepOutput, TensorDict, UnpairedAudioBatch
from src.utils import temporary_seed
import src.flop_utils

class CountFlopsExtra(ExtraMetricOutput):
    """
    A callback for profiling the FLOP cost of sampling during evaluation.
    """
    def __init__(self, key: str, out_key: str = "sampling_flops", display_table: bool = False, **kwargs):
        super().__init__()
        self.key = key
        self.out_key = out_key
        self.display_table = display_table
        self.kwargs = kwargs

    def __call__(self, pl_module: DSB, outputs: StepOutput, batch: UnpairedAudioBatch, batch_idx: int) -> TensorDict:
        #Only run on first batch, to skip redundant counting - model is not updated, so should be the same for all batches.
        if batch_idx > 0:
            return {self.out_key: 0.0} # Returns zero for everything but the first batch - to skip redundant FLOP counting

        with temporary_seed(seed=batch_idx):
            # Intercept operations happening specifically within this sample call
            with FlopCounterMode(pl_module, display=self.display_table) as flop_counter:
                _ = pl_module.sample(
                    x_start=batch[self.key],
                    **self.kwargs,
                )
                
        total_flops = tensor(float(flop_counter.get_total_flops()))
        
        return {self.out_key: total_flops}