from torch import Tensor
import torch
from src import UnpairedAudioBatch, ModelOutput, StepOutput, SchedulerBatch
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from functools import partial
from src.networks.encoders import BaseEncoderDecoder
from typing import Optional
from src.losses import BaseLossFunction
from .scheduler import DSBScheduler, DIRECTIONS, SCHEDULER_TYPES


class GFB(BaseLightningModule):
    ...