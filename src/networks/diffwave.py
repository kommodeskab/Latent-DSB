# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from torch import Tensor
import math
from typing import Optional

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, model_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, model_size, bias=True),
            nn.SiLU(),
            nn.Linear(model_size, model_size, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        t = t * 1000.0
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        dilation: int,
    ) -> None:
        """
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        """
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, diffusion_step: Tensor) -> Tensor:
        diffusion_step = self.diffusion_projection.forward(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        y = self.dilated_conv(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = self.dropout(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        residual_layers: int,
        dilation_cycle_length: int,
        num_classes: int,
    ):
        super().__init__()
        self.input_projection = Conv1d(1, residual_channels, 1)
        self.class_conditioner = nn.Embedding(num_classes, 512)
        self.diffusion_embedding = TimestepEmbedder(512)
        self.residual_layers = nn.ModuleList(
            [ResidualBlock(residual_channels, 2 ** (i % dilation_cycle_length)) for i in range(residual_layers)]
        )
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio: Tensor, diffusion_step: Tensor, class_label: Optional[Tensor] = None) -> Tensor:
        x = self.input_projection(audio)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding.forward(diffusion_step)

        if class_label is not None:
            class_embedding = self.class_conditioner(class_label)
            diffusion_step = diffusion_step + class_embedding

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer.forward(x, diffusion_step)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
