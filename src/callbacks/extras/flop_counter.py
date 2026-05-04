from torch.utils.flop_counter import FlopCounterMode
from torch import Tensor
from .base import ExtraMetricOutput
from src.lightning_modules import DSB, WPE
from src import StepOutput, UnpairedAudioBatch
from mamba_ssm import Mamba2
from src.flop_utils import calculate_mamba2_block_flops, calculate_wpe_block_flops

class CountFlopsExtra(ExtraMetricOutput):
    """
    A callback for profiling the FLOP cost of sampling during evaluation.
    """

    def __init__(self, key: str, out_key: str = "sampling_flops", display_table: bool = False, verbose=False, **kwargs):
        super().__init__()
        self.key = key
        self.out_key = out_key
        self.display_table = display_table
        self.kwargs = kwargs
        self.verbose = verbose

    def __call__(
        self, pl_module: DSB, batch: UnpairedAudioBatch, batch_idx: int
    ) -> dict[str, tuple[Tensor, list[str]]]:
        # Only run on first batch, to skip redundant counting - model is not updated, so should be the same for all batches.
        if batch_idx > 0:
            return {self.out_key: (0, 0, 0, 0, [])}
        # Returns zero for everything but the first batch - to skip redundant FLOP counting
        # Attach Mamba hooks manually
        hooks = []
        for m in pl_module.modules():
            if isinstance(m, Mamba2):
                hooks.append(m.register_forward_hook(calculate_mamba2_block_flops))
                m.__total_flops__ = 0  # reset for every hook registration

            if isinstance(m, WPE): 
                hooks.append(m.register_forward_hook(calculate_wpe_block_flops))
                m.__total_flops__ = 0

        with FlopCounterMode(display=self.display_table) as flop_counter:
            _ = pl_module.sample(
                x_start=batch[self.key],
                **self.kwargs,
            )

        torch_flops = flop_counter.get_total_flops()

        # add the manual Mamba ops
        mamba_flops = 0
        for m in pl_module.modules():
            if isinstance(m, Mamba2):
                mamba_flops += m.__total_flops__
                # print(mamba_flops)

        # add the manual WPE ops
        wpe_flops = 0
        for m in pl_module.modules():
            if isinstance(m, WPE):
                wpe_flops += m.__total_flops__

        total_flops = torch_flops + mamba_flops + wpe_flops

        # remove hooks for leaks
        for h in hooks:
            h.remove()

        # Find 0-ops for debugging
        # The counter returns a dictionary grouped by module.
        global_counts = flop_counter.get_flop_counts().get("Global", {})
        zero_flop_ops = [str(op) for op, count in global_counts.items() if count == 0]
        if self.verbose:
            print(f"\nDispatcher FLOPs (Table): {torch_flops / 1e9:.2f} Billion")
            print(f"Hidden Mamba FLOPs (Hooks): {mamba_flops / 1e9:.2f} Billion")
            print(f"Hidden WPE FLOPs (Hooks): {wpe_flops / 1e9:.2f} Billion")
            print(f"True Total: {(total_flops) / 1e9:.2f} Billion\n")

        return {self.out_key: (total_flops, torch_flops, mamba_flops, wpe_flops, zero_flop_ops)}
