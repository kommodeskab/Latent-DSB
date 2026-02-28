from typing import TypedDict, Dict, Optional
from functools import partial
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from matplotlib.figure import Figure
import numpy as np
from pytorch_lightning import LightningModule

TensorDict = Dict[str, Tensor]
OptimizerType = Optional[partial[Optimizer]]
LRSchedulerType = Optional[dict[str, partial[LRScheduler] | str]]
ImageType = list[Tensor | Figure | np.ndarray]


class AudioSample(TypedDict):
    """
    A sample from an audio dataset, containing an audio input and an optional label.
    """

    input: Tensor


class AudioBatch(TypedDict):
    """
    A batch of samples from an audio dataset, containing a
    batch of audio inputs and an optional batch of labels.
    """

    ...


class Sample(TypedDict):
    """
    A classic regression/classification sample with an input and a target.
    """

    input: Tensor
    target: Tensor


class Batch(Sample):
    """
    Same as sample but contains a batch of samples.
    """

    ...


class ModelOutput(TypedDict):
    """
    A classic output type for a model with a single output tensor.
    """

    output: Tensor


class LossOutput(TypedDict):
    """
    A classic output type for a loss function with a single loss tensor.
    """

    loss: Tensor


class StepOutput(TypedDict):
    """
    A classic output type for a training/validation/test step in PyTorch Lightning.
    * The model will take a gradient step based on `loss`,
    * The model will log the values in `loss_output` (which can contain multiple values,
    e.g. a VAE loss might contain a reconstruction loss and a KL divergence loss) if and only
    if you are using the `LogLossCallback`. See the `LogLossCallback` for more details.
    """

    loss: Tensor
    model_output: Optional[ModelOutput] = None
    loss_output: Optional[LossOutput] = None
    module: Optional[LightningModule] = None
