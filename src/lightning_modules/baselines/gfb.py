import torch
import numpy as np
import einops
import ot as pot
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import os
from pathlib import Path
from types import SimpleNamespace
import requests
from typing import Literal
from torch import Tensor
from src.lightning_modules import BaseLightningModule


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=1,
        bias=False,
        dilation=1,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel, fan_out=out_channels * kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if w is not None:
            x = torch.nn.functional.conv1d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))
        return x


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(1, 1),
        bias=False,
        dilation=1,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel[0] * kernel[1],
            fan_out=out_channels * kernel[0] * kernel[1],
        )
        self.weight = torch.nn.Parameter(
            weight_init([out_channels, in_channels, kernel[0], kernel[1]], **init_kwargs) * init_weight
        )
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if w is not None:
            x = torch.nn.functional.conv2d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class BiasFreeGroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-7):
        super(BiasFreeGroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, F, T = x.size()
        gc = C // self.num_groups
        x = einops.rearrange(x, "n (g gc) f t -> n g (gc f t)", g=self.num_groups, gc=gc)

        std = x.std(-1, keepdim=True)  # reduce over channels and time

        ## normalize
        x = (x) / (std + self.eps)
        # normalize
        x = einops.rearrange(x, "n g (gc f t) -> n (g gc) f t", g=self.num_groups, gc=gc, f=F, t=T)
        return x * self.gamma


class RFF_MLP_Block(nn.Module):
    """
    Encoder of the noise level embedding
    Consists of:
        -Random Fourier Feature embedding
        -MLP
    """

    def __init__(self, emb_dim=512, rff_dim=32, inputs=1, init=None):
        super().__init__()
        self.inputs = inputs
        self.RFF_freq = nn.Parameter(16 * torch.randn([1, rff_dim]), requires_grad=False)
        self.MLP = nn.ModuleList(
            [
                Linear(2 * rff_dim * self.inputs, 128, **init),
                Linear(128, 256, **init),
                Linear(256, emb_dim, **init),
            ]
        )

    def forward(self, x):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = [self._build_RFF_embedding(x[:, i].unsqueeze(-1)) for i in range(self.inputs)]
        x = torch.cat(x, dim=1)

        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=True,
        num_dils=6,
        bias=False,
        kernel_size=(5, 3),
        emb_dim=512,
        proj_place="before",  # using 'after' in the decoder out blocks
        init=None,
        init_zero=None,
    ):
        super().__init__()

        if emb_dim == 0:
            self.no_emb = True
        else:
            self.no_emb = False

        self.bias = bias
        self.use_norm = use_norm
        self.num_dils = num_dils
        self.proj_place = proj_place

        if self.proj_place == "before":
            # dim_out is the block dimension
            N = dim_out
        else:
            # dim in is the block dimension
            N = dim
            self.proj_out = (
                Conv2d(N, dim_out, bias=bias, **init) if N != dim_out else nn.Identity()
            )  # linear projection

        self.res_conv = (
            Conv2d(dim, dim_out, bias=bias, **init) if dim != dim_out else nn.Identity()
        )  # linear projection
        self.proj_in = Conv2d(dim, N, bias=bias, **init) if dim != N else nn.Identity()  # linear projection

        self.H = nn.ModuleList()
        self.affine = nn.ModuleList()
        self.gate = nn.ModuleList()
        if self.use_norm:
            self.norm = nn.ModuleList()

        for i in range(self.num_dils):
            if self.use_norm:
                self.norm.append(BiasFreeGroupNorm(N, 8))

            if not self.no_emb:
                self.affine.append(Linear(emb_dim, N, **init))
                self.gate.append(Linear(emb_dim, N, **init_zero))
            # freq convolution (dilated)
            self.H.append(Conv2d(N, N, kernel=kernel_size, dilation=(2**i, 1), bias=bias, **init))

    def forward(self, input_x, sigma):
        x = input_x

        x = self.proj_in(x)

        if self.no_emb:
            for norm, conv in zip(self.norm, self.H):
                x0 = x
                if self.use_norm:
                    x = norm(x)
                x = (x0 + conv(F.gelu(x))) / (2**0.5)
        else:
            for norm, affine, gate, conv in zip(self.norm, self.affine, self.gate, self.H):
                x0 = x
                if self.use_norm:
                    x = norm(x)
                gamma = affine(sigma)
                scale = gate(sigma)

                x = x * (gamma.unsqueeze(2).unsqueeze(3) + 1)  # no bias

                x = (x0 + conv(F.gelu(x)) * scale.unsqueeze(2).unsqueeze(3)) / (2**0.5)

        # one residual connection here after the dilated convolutions

        if self.proj_place == "after":
            x = self.proj_out(x)

        x = (x + self.res_conv(input_x)) / (2**0.5)

        return x


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [
        -0.01171875,
        -0.03515625,
        0.11328125,
        0.43359375,
        0.43359375,
        0.11328125,
        -0.03515625,
        -0.01171875,
    ],
}


class UpDownResample(nn.Module):
    def __init__(
        self,
        up=False,
        down=False,
        mode_resample="T",  # T for time, F for freq, TF for both
        resample_filter="cubic",
        pad_mode="reflect",
    ):
        super().__init__()
        assert not (up and down)  # you cannot upsample and downsample at the same time
        assert up or down  # you must upsample or downsample
        self.down = down
        self.up = up
        if up or down:
            # upsample block
            self.pad_mode = pad_mode  # I think reflect is a goof choice for padding
            self.mode_resample = mode_resample
            if mode_resample == "T":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            elif mode_resample == "F":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            else:
                raise NotImplementedError("Only time upsampling is implemented")
            self.pad = kernel_1d.shape[0] // 2 - 1
            self.register_buffer("kernel", kernel_1d)

    def forward(self, x):
        shapeorig = x.shape
        x = x.view(-1, x.shape[-2], x.shape[-1])
        if self.mode_resample == "F":
            x = x.permute(0, 2, 1)

        if self.down:
            x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        elif self.up:
            x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)

        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)

        weight[indices, indices] = self.kernel.to(weight)
        if self.down:
            x_out = F.conv1d(x, weight, stride=2)
        elif self.up:
            x_out = F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)

        if self.mode_resample == "F":
            x_out = x_out.permute(0, 2, 1).contiguous()
            return x_out.view(shapeorig[0], -1, x_out.shape[-2], shapeorig[-1])
        else:
            return x_out.view(shapeorig[0], -1, shapeorig[2], x_out.shape[-1])


class STFTbackbone(nn.Module):
    """
    Main U-Net model based on the STFT
    """

    def __init__(
        self,
        stft_args=SimpleNamespace(
            win_length=510,
            hop_length=128,
        ),
        depth=7,
        emb_dim=256,
        use_norm=True,
        Ns=[64, 128, 256, 512, 512, 512, 512],
        Ss=[2, 2, 2, 2, 2, 2, 2],
        num_dils=[1, 1, 1, 1, 1, 1, 1],
        bottleneck_type="res_dil_convs",
        num_bottleneck_layers=1,
        device="cuda",
        time_conditional=True,
        param_conditional=True,
        num_cond_params=2,
        output_channels=1,
    ):
        """
        Args:
            args (dictionary): hydra dictionary
            device: torch device ("cuda" or "cpu")
        """
        super(STFTbackbone, self).__init__()
        self.stft_args = stft_args
        self.win_size = stft_args.win_length
        self.hop_size = stft_args.hop_length

        self.time_conditional = time_conditional
        self.param_conditional = param_conditional
        self.num_cond_params = num_cond_params
        self.depth = depth

        init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3))
        init_zero = dict(init_mode="kaiming_uniform", init_weight=1e-7)

        self.emb_dim = emb_dim
        self.total_emb_dim = 0
        if self.time_conditional:
            self.embedding = RFF_MLP_Block(emb_dim=emb_dim, inputs=1, init=init)
            self.total_emb_dim += emb_dim
        if self.param_conditional:
            self.embedding_param = RFF_MLP_Block(emb_dim=emb_dim, inputs=num_cond_params, init=init)
            self.total_emb_dim += emb_dim

        self.use_norm = use_norm

        self.device = device

        Nin = 2

        Nout = 2 * output_channels

        # Encoder
        self.Ns = Ns
        self.Ss = Ss

        self.num_dils = num_dils

        self.downsamplerT = UpDownResample(down=True, mode_resample="T")
        self.downsamplerF = UpDownResample(down=True, mode_resample="F")
        self.upsamplerT = UpDownResample(up=True, mode_resample="T")
        self.upsamplerF = UpDownResample(up=True, mode_resample="F")

        self.downs = nn.ModuleList([])
        self.middle = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.init_block = ResnetBlock(
            Nin,
            self.Ns[0],
            self.use_norm,
            num_dils=1,
            bias=False,
            kernel_size=(1, 1),
            emb_dim=self.total_emb_dim,
            init=init,
            init_zero=init_zero,
        )
        self.out_block = ResnetBlock(
            self.Ns[0],
            Nout,
            use_norm=self.use_norm,
            num_dils=1,
            bias=False,
            kernel_size=(1, 1),
            proj_place="after",
            emb_dim=self.total_emb_dim,
            init=init,
            init_zero=init_zero,
        )

        for i in range(self.depth):
            if i == 0:
                dim_in = self.Ns[i]
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i - 1]
                dim_out = self.Ns[i]

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_out,
                            self.use_norm,
                            num_dils=self.num_dils[i],
                            bias=False,
                            emb_dim=self.total_emb_dim,
                            init=init,
                            init_zero=init_zero,
                        ),
                    ]
                )
            )

        self.bottleneck_type = bottleneck_type
        self.num_bottleneck_layers = num_bottleneck_layers
        if self.bottleneck_type == "res_dil_convs":
            for i in range(self.num_bottleneck_layers):
                self.middle.append(
                    nn.ModuleList(
                        [
                            ResnetBlock(
                                self.Ns[-1],
                                self.Ns[-1],
                                self.use_norm,
                                num_dils=self.num_dils[-1],
                                bias=False,
                                emb_dim=self.total_emb_dim,
                                init=init,
                                init_zero=init_zero,
                            )
                        ]
                    )
                )
        else:
            raise NotImplementedError("bottleneck type not implemented")

        for i in range(self.depth - 1, -1, -1):
            if i == 0:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i - 1]

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_out,
                            use_norm=self.use_norm,
                            num_dils=self.num_dils[i],
                            bias=False,
                            emb_dim=self.total_emb_dim,
                            init=init,
                            init_zero=init_zero,
                        ),
                    ]
                )
            )

    def forward_backbone(self, inputs, time_cond=None, param_cond=None):
        """
        Args:
            inputs (Tensor):  Input signal in frequency-domsin, shape (B,C,F,T)
            sigma (Tensor): noise levels,  shape (B,1)
        Returns:
            pred (Tensor): predicted signal in time-domain, shape (B,C,F,T)
        """
        # apply RFF embedding+MLP of the noise level
        emb = None
        if self.time_conditional:
            time_cond = time_cond.unsqueeze(-1)
            time_cond = self.embedding(time_cond)
            if not self.param_conditional:
                emb = time_cond

        if self.param_conditional:
            param_cond = param_cond
            param_cond = self.embedding_param(param_cond)
            if not self.time_conditional:
                emb = param_cond
            else:
                emb = torch.cat((time_cond, param_cond), dim=1)

        hs = []

        X = self.init_block(inputs, emb)

        for i, modules in enumerate(self.downs):
            (ResBlock,) = modules

            X = ResBlock(X, emb)
            hs.append(X)

            # downsample the main signal path
            # we do not need to downsample in the inner layer
            if i < len(self.downs) - 1:
                # no downsampling in the last layer
                X = self.downsamplerT(X)
                X = self.downsamplerF(X)

        # middle layers
        if self.bottleneck_type == "res_dil_convs":
            for i in range(self.num_bottleneck_layers):
                (ResBlock,) = self.middle[i]
                X = ResBlock(X, emb)

        for i, modules in enumerate(self.ups):
            j = len(self.ups) - i - 1

            (ResBlock,) = modules

            skip = hs.pop()
            # print("skip", skip.shape)
            X = torch.cat((X, skip), dim=1)
            X = ResBlock(X, emb)
            if j > 0:
                # no upsampling in the first layer
                X = self.upsamplerT(X)  # call contiguous() here?
                X = self.upsamplerF(X)  # call contiguous() here?

        X = self.out_block(X, emb)

        return X

    def do_stft(self, x):
        """
        x shape: (batch, C, time)
        """
        window = torch.hamming_window(window_length=self.win_size, device=x.device)

        x = torch.cat(
            (
                x,
                torch.zeros((x.shape[0], x.shape[1], self.win_size - 1), device=x.device),
            ),
            -1,
        )
        B, C, T = x.shape
        x = x.view(-1, x.shape[-1])
        stft_signal = torch.stft(
            x,
            self.win_size,
            hop_length=self.hop_size,
            window=window,
            center=False,
            return_complex=True,
        )
        stft_signal = torch.view_as_real(stft_signal)

        stft_signal = stft_signal.view(B, C, *stft_signal.shape[1:])
        # shape (batch, C, freq, time, 2)

        return stft_signal

    def do_istft(self, x):
        """
        x shape: (batch, C, freq, time, 2)
        """
        B, C, F, T, _ = x.shape
        x = torch.view_as_complex(x)
        window = torch.hamming_window(window_length=self.win_size, device=x.device)  # this is slow! consider optimizing
        x = einops.rearrange(x, "b c f t -> (b c) f t ")
        pred_time = torch.istft(
            x,
            self.win_size,
            hop_length=self.hop_size,
            window=window,
            center=False,
            return_complex=False,
        )
        pred_time = einops.rearrange(pred_time, "(b c) t -> b c t", b=B)
        return pred_time

    def forward(self, x, time_cond=None, cond=None):
        B, C, T = x.shape
        # apply stft
        x = self.do_stft(x)

        x = einops.rearrange(x, "b c f t ri -> b (c ri) f t")
        x = self.forward_backbone(x, time_cond, cond)
        # apply istft
        x = einops.rearrange(x, " b (c ri) f t -> b c f t ri", ri=2)
        x = x.contiguous()
        x = self.do_istft(x)
        x = x[:, :, :T]
        return x


def calculate_curvature(trajectory):
    # as used in the paper, just for reference
    base = trajectory[0] - trajectory[-1]
    base = base.reshape(base.shape[0], -1)
    N = len(trajectory)
    dt = 1.0 / N
    mse = []
    for i in range(1, N):
        v = (trajectory[i - 1] - trajectory[i]) / dt
        v = v.reshape(v.shape[0], -1)
        mse.append(torch.mean((v - base) ** 2, dim=-1).cpu())
    return torch.mean(torch.stack(mse)), mse


class OTCFM:
    """
    Class that takes care of all diffusion-related stuff. This includes training losses, sampling, etc.
    """

    def __init__(
        self,
        cfg_value=2,
        num_cond_params=2,
        minibatch_OT=False,
        minibatch_OT_args=None,
        order=1,
    ) -> None:
        self.cfg_value = cfg_value
        self.num_cond_params = num_cond_params
        self.minibatch_OT = minibatch_OT
        self.minibatch_OT_args = minibatch_OT_args
        self.order = order
        if self.minibatch_OT:
            if self.minibatch_OT_args.algorithm == "emd":
                self.ot_fn = partial(pot.emd, numItermax=1e5, numThreads=1)
            elif self.minibatch_OT_args.algorithm == "sinkhorn":
                self.ot_fn = partial(pot.sinkhorn, reg=0.001, numItermax=int(1e2), method="sinkhorn_log")

    def get_train_tuple(self, x, z, t):
        # linear interpolation
        x_t = t * z + (1.0 - t) * x

        # CFM objective
        target = z - x
        return x_t, target

    def OT_plan(self, x, z):
        # divide the pairwise L2 operation to save memory
        # compute pairwise L2 matrix
        M = ((x[None, :, :] - z[:, None, :]) ** 2).mean(-1)

        M = M.T
        M += 1e-5  # for numerical stability

        # assert that M is a square matrix
        assert M.shape[0] == M.shape[1]
        a, b = pot.unif(M.shape[0]), pot.unif(M.shape[1])
        a = torch.from_numpy(a).float().to(M.device)
        b = torch.from_numpy(b).float().to(M.device)

        # apply the solver
        P = self.ot_fn(a, b, M)

        if self.minibatch_OT_args.algorithm == "emd":
            # in the case of deterministic OT, this is equivalent to what we do below, and probably faster
            index = P.max(axis=1).indices
        else:
            P *= x.shape[0]
            P /= P.sum(-1)
            normalized_P = P / P.sum(dim=1, keepdim=True)
            index = torch.multinomial(normalized_P, 1, replacement=True).squeeze()

        return index

    def get_audio_noise_pairs(self, x, x_plan):
        B, C, T = x.shape
        assert C == 1, "C must be 1"
        # divide the audio in chunks
        x_chunk = einops.rearrange(x_plan.squeeze(1), "b (t c) -> (b t) c", c=self.minibatch_OT_args.chunk_size)

        z = torch.randn((x_chunk.shape[0], self.minibatch_OT_args.chunk_size), device=x.device)

        index = self.OT_plan(x_chunk, z)
        z = z[index]

        z = einops.rearrange(z, "(b t) c -> b (t c)", b=B, c=self.minibatch_OT_args.chunk_size)

        z = z.unsqueeze(1)
        return x, z

    def compute_loss(self, x, model=None, cond=None, task="reverb"):
        assert model is not None, "model must be provided"

        loss_dict = {}

        if task == "reverb":
            T60 = cond["T60"]
            C50 = cond["C50"]
            used_conds = [T60, C50]
        elif task == "declipping":
            SDR = cond["SDR"]
            used_conds = [SDR]
        else:
            raise NotImplementedError("the task {} is not implemented".format(task))

        if self.minibatch_OT:
            x_plan = x
            x, z = self.get_audio_noise_pairs(x, x_plan)
        else:
            # classic training with independent noise
            z = torch.randn_like(x)

        B = x.shape[0]
        t = torch.rand((B), device=x.device)
        eps = 1e-6
        t = t * (1 - eps) + eps
        # t must lie in (0,1)
        assert torch.all(t > 0) and torch.all(t < 1)

        x_t, target = self.get_train_tuple(x, z, t.unsqueeze(1).unsqueeze(1))

        if cond is not None:
            cond_tensor = [c.to(x.device).to(x.dtype).unsqueeze(-1) for c in used_conds]
            # concatenate both in a single tensor
            cond_tensor = torch.cat(cond_tensor, dim=1)
            # with a certain probability, set all the condition to some predefined value
            dropped = torch.rand(cond_tensor.shape, dtype=cond_tensor.dtype, device=cond_tensor.device) < 0.8
            cond_tensor = cond_tensor * dropped.float() + self.cfg_value * (1 - dropped.float())

        if len(cond_tensor.shape) == 3:
            cond_tensor = cond_tensor.squeeze(-1)
        if len(t) == 1:
            t = t.unsqueeze(-1)
        pred = model(x_t, torch.log(t), cond=cond_tensor)

        loss = (target.clone() - pred) ** 2
        loss_dict["error"] = loss.detach()

        total_loss = loss.mean()

        return total_loss, loss_dict, t

    def get_schedule(self, Tsteps, end_t=1, type="linear"):
        if type == "linear":
            return torch.linspace(0, end_t, Tsteps + 1)[1:]
        elif type == "cosine":
            pi = torch.tensor(np.pi)
            # if end_t!=1:
            #    raise NotImplementedError("end_t must be 1 for cosine schedule")
            t = torch.linspace(0, 1, Tsteps + 1)
            base = 0.5 * (1 + torch.cos(t * pi + pi))

            # cut it at end_t (find the index of the first element that is greater than end_t)
            base = base[base <= end_t]

            return base[1:]

    def sample_conditional(
        self,
        shape,  # B, C, T
        model,  # DNN
        Tsteps,  # number of steps
        cond=None,  # conditioning parameters
        cfg=None,  # classifier-free guidance
        same_noise=False,  # if True, use the same noise for all the elements of the batch
        device="cuda",
    ):
        B, C, T = shape
        t = self.get_schedule(Tsteps).to(device)
        # sample prior
        if not same_noise:
            z = torch.randn(B, C, T).to(device)
        else:
            z = torch.randn(1, C, T).to(device)
            z = z.expand(B, C, T)

        xt = z
        for i in reversed(range(0, Tsteps)):
            if i == 0:
                xt = self.sampling_step(
                    xt,
                    t[i].expand(
                        B,
                    ),
                    t[i].expand(
                        B,
                    )
                    * 0,
                    model,
                    1,
                    cond=cond,
                    cfg=cfg,
                )
            else:
                xt = self.sampling_step(
                    xt,
                    t[i].expand(
                        B,
                    ),
                    t[i - 1].expand(
                        B,
                    ),
                    model,
                    self.order,
                    cond=cond,
                    cfg=cfg,
                )
        return xt

    def sample_unconditional(
        self,
        shape,  # B, C, T
        model,  # DNN
        Tsteps,  # number of steps
        device,
        output_trajectory=False,
    ):
        B, C, T = shape
        t = self.get_schedule(Tsteps).to(device)
        # sample prior
        z = torch.randn(B, C, T).to(device) * t[-1]

        trajectory = []
        denoised_estimates = []
        xt = z
        trajectory.append(xt)
        for i in reversed(range(0, Tsteps)):
            if i == 0:
                xt, x0 = self.sampling_step(
                    xt,
                    t[i].expand(
                        B,
                    ),
                    t[i].expand(
                        B,
                    )
                    * 0,
                    model,
                    1,
                    get_denoised_estimate=True,
                )
                trajectory.append(xt)
                denoised_estimates.append(x0)
            else:
                xt, x0 = self.sampling_step(
                    xt,
                    t[i].expand(
                        B,
                    ),
                    t[i - 1].expand(
                        B,
                    ),
                    model,
                    self.order,
                    get_denoised_estimate=True,
                )
                trajectory.append(xt)
                denoised_estimates.append(x0)
        if output_trajectory:
            return xt, trajectory, denoised_estimates
        else:
            return xt

    def model_call(self, model, x, t, cond=None):
        if cond is None:
            cond = torch.zeros((x.shape[0], self.num_cond_params), dtype=x.dtype, device=x.device) + self.cfg_value
        with torch.no_grad():
            v = model(x, torch.log(t), cond=cond)
        return v

    def model_call_cfg(self, model, x, t, cond=None, cfg=None):
        v_unc = self.model_call(model, x, t)
        v_cond = self.model_call(model, x, t, cond=cond)
        return (1 - cfg) * v_unc + cfg * v_cond

    def sampling_step(
        self,
        xt,  # noisy input
        t,  # current timestep
        t2,  # next timestep
        model,  # DNN
        order=1,  # 1 for Euler, 2 for Heun
        cond=None,  # conditioning parameters
        cfg=None,  # classifier-free guidance
        get_denoised_estimate=False,  # if True, return the denoised estimate
    ):
        if cond is not None:
            assert cfg is not None, "If cond is not None, cfg must be provided"
        if cond is None:
            vt = self.model_call(model, xt, t)
        else:
            vt = self.model_call_cfg(model, xt, t, cond=cond, cfg=cfg)

        # using
        dt = (t2 - t).view(-1, 1, 1)
        # print(dt)

        if order == 2:
            x2 = xt.detach().clone() + vt * dt
            if cond is None:
                vt_2 = self.model_call(model, x2, t2)
            else:
                vt_2 = self.model_call_cfg(model, x2, t2, cond=cond, cfg=cfg)
            vt = (vt + vt_2) / 2

        if get_denoised_estimate:
            denoised_estimate = xt.detach().clone() - vt * t.view(-1, 1, 1)

        xt = xt.detach().clone() + vt * dt

        if get_denoised_estimate:
            return xt, denoised_estimate
        else:
            return xt

    def bridge(
        self,
        x0,
        model=None,
        Tsteps=30,
        cond=None,
        cfg=None,
        bridge_end_t=1,
        output_trajectory=False,
        schedule_type="linear",
    ):
        """
        Diffusion bridge, where we apply unconditonal forward diffusion, and conditional backward diffusion
        args:
            x0: (1, C, T) tensor
            Tsteps: number of steps
            cond: (1, num_cond_params) tensor
        """
        assert model is not None, "model must be provided"
        device = x0.device

        t_schedule = self.get_schedule(Tsteps, end_t=bridge_end_t, type=schedule_type).to(device)
        if output_trajectory:
            trajectory = {}
            trajectory["t"] = t_schedule
        # sample prior
        # maybe start with a small sigma
        x0 = t_schedule[0] * torch.randn(x0.shape, device=device) + (1.0 - t_schedule[0]) * x0
        if output_trajectory:
            z, traj = self.forward_ODE(x0, model=model, schedule=t_schedule, output_trajectory=output_trajectory)
            trajectory["forward"] = traj
        else:
            z = self.forward_ODE(x0, model=model, schedule=t_schedule, output_trajectory=output_trajectory)

        if cond is not None:
            B = cond.shape[0]
        else:
            B = x0.shape[0]
        # expand z to B
        zexp = z.expand(B, -1, -1)
        if output_trajectory:
            xnew, traj = self.backward_ODE(
                zexp, model=model, cond=cond, cfg=cfg, schedule=t_schedule, output_trajectory=output_trajectory
            )
            trajectory["backward"] = traj
        else:
            xnew = self.backward_ODE(
                zexp, model=model, cond=cond, cfg=cfg, schedule=t_schedule, output_trajectory=output_trajectory
            )

        if output_trajectory:
            return xnew, z, trajectory
        else:
            return xnew, z

    def forward_ODE(
        self,
        xt,  # B, C, T
        model=None,  # DNN
        cond=None,  # conditioning parameters
        cfg=None,  # classifier-free guidance
        schedule=None,  # schedule of timesteps
        output_trajectory=False,
    ):
        B, C, T = xt.shape
        Tsteps = schedule.shape[0]

        trajectory = []
        trajectory.append(xt)
        for i in range(0, Tsteps):
            if i == Tsteps - 1:
                xt = self.sampling_step(
                    xt,
                    schedule[i].expand(
                        B,
                    ),
                    schedule[i].expand(
                        B,
                    )
                    * 0
                    + schedule[-1],
                    model,
                    1,
                    cond=cond,
                    cfg=cfg,
                )
                trajectory.append(xt)
            else:
                xt = self.sampling_step(
                    xt,
                    schedule[i].expand(
                        B,
                    ),
                    schedule[i + 1].expand(
                        B,
                    ),
                    model,
                    2,
                    cond=cond,
                    cfg=cfg,
                )
                trajectory.append(xt)

        if output_trajectory:
            return xt, trajectory
        else:
            return xt

    def backward_ODE(
        self,
        xt,  # B, C, T
        model=None,  # DNN
        cond=None,  # conditioning parameters
        cfg=None,  # classifier-free guidance
        schedule=None,  # schedule of timesteps
        output_trajectory=False,
    ):
        B, C, T = xt.shape
        Tsteps = schedule.shape[0]
        trajectory = []
        trajectory.append(xt)
        for i in reversed(range(0, Tsteps)):
            if i == 0:
                xt = self.sampling_step(
                    xt,
                    schedule[i].expand(
                        B,
                    ),
                    schedule[i].expand(
                        B,
                    )
                    * 0,
                    model,
                    1,
                    cond=cond,
                    cfg=cfg,
                )
                trajectory.append(xt)
            else:
                xt = self.sampling_step(
                    xt,
                    schedule[i].expand(
                        B,
                    ),
                    schedule[i - 1].expand(
                        B,
                    ),
                    model,
                    self.order,
                    cond=cond,
                    cfg=cfg,
                )
                trajectory.append(xt)

        if output_trajectory:
            return xt, trajectory
        else:
            return xt


def download_file(url: str, target_folder: str):
    target_dir = Path(target_folder)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = url.split("/")[-1]
    destination = target_dir / file_name

    if destination.exists():
        return destination

    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for HTTP errors
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return destination


class GFB(BaseLightningModule):
    """
    Gaussian Flow Bridges for audio domain transfer with unpaired data
    Taken from here:
    https://github.com/microsoft/GFB-audio-control

    Used as a baseline.
    """

    def __init__(
        self,
        task: Literal["rir", "clip"],
    ):
        super().__init__()

        load_dotenv()

        data_path = os.getenv("DATA_PATH")

        urls = {
            "rir": "https://github.com/microsoft/GFB-audio-control/releases/download/public_weights/checkpoint_299999_speech_reverb_C-OT_NC128.pt",
            "clip": "https://github.com/microsoft/GFB-audio-control/releases/download/public_weights/checkpoint_299999_speech_clipping_IndepCoupling.pt",
        }

        self.task = task

        num_conds = 2 if task == "rir" else 1
        self.model = STFTbackbone(num_cond_params=num_conds)
        self.diffusion = OTCFM(
            cfg_value=-999,
            minibatch_OT=False,
            order=2,
            minibatch_OT_args={
                "chunk_size": 256,
                "n_noise_samples": 8192,
                "distance": "l2",
                "dist_compute_num": 1024,
                "algorithm": "sinkhorn",
            },
        )
        url = urls[self.task]
        path = download_file(url, data_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def common_step(self, batch, batch_idx):
        return ...

    @torch.no_grad()
    def sample(self, x_start: Tensor, num_steps: int, **kwargs) -> Tensor:
        # normalize each sample in the batch to be between -1 and 1
        # x_start.shape = (B, C, T)
        # normalize each sample in the batch to be between -1 and 1
        # x_start = x_start / (x_start.abs().max(dim=-1, keepdim=True)[0] + 1e-5)

        sigma_data = 0.05
        x_start = x_start / sigma_data

        self.model.eval()
        B, C, T = x_start.shape

        if self.task == "rir":
            # full denoising:
            # T60=0, C50=50
            cond = torch.tensor([[0.0, 50.0]], device=x_start.device).repeat(B, 1)
        elif self.task == "clip":
            # full declipping:
            # SDR = 50, which is a very high value, corresponding to removing clipping
            cond = torch.tensor([[50.0]], device=x_start.device).repeat(B, 1)

        out, _ = self.diffusion.bridge(
            x0=x_start,
            model=self.model,
            Tsteps=num_steps // 2,
            cond=cond,
            cfg=2.0,
            schedule_type="linear",
            bridge_end_t=1.0,
        )

        return out * sigma_data
