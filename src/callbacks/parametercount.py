from pytorch_lightning import Callback
import pytorch_lightning as pl
import torch.nn as nn
import wandb
from src.lightning_modules.baselightningmodule import BaseLightningModule


class ParameterCountCallback(Callback):
    """
    Count and log the number of parameters in the model at the start of training.
    Logs both the paramter count for the full model and for its network component.
    """

    def __init__(self, depth: int = 1):
        super().__init__()
        self.depth = depth

    def count_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def collect_counts(self, module: nn.Module) -> dict[int, dict[str, int]]:
        modules = [("total", module, 0)]  # list of (name, module, current_depth)
        counts = {}

        while modules:
            name, mod, depth = modules.pop()
            if depth > self.depth:
                continue

            if depth not in counts:
                counts[depth] = {}

            counts[depth][name] = self.count_params(mod)

            for child_name, child_mod in mod.named_children():
                full_name = f"{name}.{child_name}"
                modules.append((full_name, child_mod, depth + 1))

        return counts

    def on_train_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        params = self.collect_counts(pl_module)

        # log a bar plot for each depth level
        for depth, modules in params.items():
            # Convert depth dict to a list of lists for the W&B plotter
            plot_data = [[name, count] for name, count in modules.items()]
            table = wandb.Table(data=plot_data, columns=["module", "parameters"])

            # Log a specific bar chart for this depth
            wandb.log(
                {
                    f"Parameters/Depth {depth}": wandb.plot.bar(
                        table, "module", "parameters", title=f"Parameters at depth {depth}"
                    )
                }
            )
