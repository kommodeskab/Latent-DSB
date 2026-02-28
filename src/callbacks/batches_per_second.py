from pytorch_lightning import Callback
import pytorch_lightning as pl
import time
from typing import Literal


class BatchesPerSecondCallback(Callback):
    """
    A callback for logging the number of batches processed per second.
    This can be useful for monitoring training performance.
    Logs both during training, validation, and testing phases.
    """

    def __init__(self):
        # initialize time trackers
        self.info = {
            "train": {"running_avg": None, "last_time": None},
            "val": {"running_avg": None, "last_time": None},
            "test": {"running_avg": None, "last_time": None},
        }
        self.alpha = 0.01

    def _track_bps(self, phase: Literal["train", "val", "test"], pl_module: pl.LightningModule):
        info = self.info[phase]
        before = info["last_time"]
        now = time.time()

        if before is None:
            # if this is the first batch, just set the time
            info["last_time"] = now
            return

        elapsed = now - before
        bps = 1.0 / elapsed
        running_avg = info["running_avg"]
        updated_avg = bps if running_avg is None else (1 - self.alpha) * running_avg + self.alpha * bps
        pl_module.log(f"{phase}_batches_per_second", updated_avg, sync_dist=(phase != "train"))

        info["running_avg"] = updated_avg
        info["last_time"] = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # this measures time between the start of batches, i.e. a full iteration
        # could use almost any hook instead, so this is arbitrary
        self._track_bps("train", pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._track_bps("val", pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._track_bps("test", pl_module)
