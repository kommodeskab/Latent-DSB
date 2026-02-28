from torchview import draw_graph
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from PIL import Image
import io


class LogGraphCallback(Callback):
    def __init__(
        self,
        keyword: str,
        input_shape: list[tuple[int, ...]],
        depth: int = 2,
    ):
        super().__init__()
        self.keyword = keyword
        self.input_shape = input_shape
        self.depth = depth

    def on_train_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        if hasattr(self, "has_logged"):
            return

        inputs = []
        for shape in self.input_shape:
            shape = list(shape)
            inputs.append(torch.randn(shape).to(pl_module.device))

        network = getattr(pl_module, self.keyword)
        graph = draw_graph(network, inputs, depth=self.depth, save_graph=False)
        png_data = graph.visual_graph.pipe(format="png")

        pil_image = Image.open(io.BytesIO(png_data))
        pl_module.logger.log_image(key="model_graph", images=[pil_image])

        self.has_logged = True
