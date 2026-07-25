import torch
from torch.utils.flop_counter import register_flop_formula


def calculate_mamba2_block_flops(module, input, output):
    """
    Analytically calculates missing FLOPs for Mamba2 blocks (Scan + Conv1d)
    and attaches them to the module.
    See reference https://github.com/state-spaces/mamba/issues/110 for discussion and
    https://arxiv.org/abs/2405.21060 for paper implementation (and thus differences from above discussion)
    """
    x = input[0]
    batch, seq_len, d_model = x.shape

    # Extract structural dimensions dynamically for Mamba-2
    d_inner = module.d_inner
    n_heads = module.nheads
    d_state = module.d_state
    chunk_size = module.chunk_size
    d_conv = module.d_conv
    d_ssm = module.d_ssm
    ngroups = module.ngroups

    assert module.use_mem_eff_path, "This flop counter assumes use_mem_eff_path = True"
    # Mamba-2 convolves over x, B, and C, so the channel dim is slightly larger than d_inner
    conv_dim = module.conv1d.in_channels

    # Count for silu activations and conv as they are hidden to torch.
    # They are added in the same call, see line 840 in ssd_combined in mamba_ssm
    conv_flops = 2 * batch * seq_len * conv_dim * d_conv
    silu_flops = 5 * batch * seq_len * conv_dim

    # below is a gated norm, see ssd_combined in mamba_ssm, line 868
    rmsnorm_flops = 11 * batch * seq_len * d_ss

    hidden_flops = conv_flops + silu_flops + rmsnorm_flops

    # NOTE: If d_ssm < d_inner, Mamba-2 creates an extra MLP branch (d_mlp).
    # If use_mem_eff_path=True, this branch uses a hidden Triton SwiGLU kernel.
    d_mlp = (module.in_proj.out_features - 2 * d_ssm - 2 * ngroups * d_state - n_heads) // 2
    if d_mlp > 0:
        print(d_mlp)
        # Hidden ops: SiLU (5) + Multiply (1) = 6 ops on d_mlp
        hidden_flops += 6 * batch * seq_len * d_mlp

    # Mamba-2 SSD Combined Scan FLOPs (Diagonal blocks + Low-rank blocks)
    # NOTE This differs from the github reference, since that was for mamba, not mamba2.
    # See chapter 6 of the paper for this analysis.
    scan_flops = 2 * batch * seq_len * d_state * ((n_heads * chunk_size) + (3 * d_inner))

    # Calculate element-wise multiplication flops
    mask_flops = batch * seq_len * n_heads * chunk_size

    # Accumulate
    if not hasattr(module, "__total_flops__"):
        module.__total_flops__ = 0
    module.__total_flops__ += scan_flops + hidden_flops + mask_flops


def calculate_wpe_block_flops(module, input, output):
    """
    Analytically calculates hidden FLOPs for the NARA-WPE algorithm.
    Dynamically adapts to the STFT frame count, including fading pads.
    """
    x = input[0]
    B, D, L = x.shape
    stft_size = module.stft_size
    stft_shift = module.stft_shift
    K = module.taps
    iterations = module.iterations
    # Calculate the resulting STFT dimensions
    F = (stft_size // 2) + 1
    # nara_wpe default fading=True pads both sides by (size - shift)
    pad_width = stft_size - stft_shift
    padded_L = L + 2 * pad_width
    # Calculate exact number of STFT time frames (T)
    T = (padded_L + stft_shift - stft_size) // stft_shift
    # print(B,F,D,T)

    K = module.taps
    iterations = module.iterations

    # Section 4.1 of the paper describes the algorithm.
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578026&tag=1
    # Covariance Matrix (R)
    r_flops = 8 * F * T * (D * K) ** 2

    # Cross-Correlation Vector (P)
    p_flops = 8 * F * T * (D**2 * K)

    # Matrix Solve (G = R^-1 P)
    solve_flops = F * ((8 / 3) * (D * K) ** 3 + 8 * (D * K) ** 2 * D)

    # Filtering
    filter_flops = 8 * F * T * (D**2 * K)

    # Calculate Total FLOPs
    flops_per_batch = iterations * (r_flops + p_flops + solve_flops + filter_flops)
    total_flops = B * flops_per_batch

    # Accumulate
    if not hasattr(module, "__total_flops__"):
        module.__total_flops__ = 0
    module.__total_flops__ += total_flops


@register_flop_formula(torch.ops.aten.silu, get_raw=True)
@register_flop_formula(torch.ops.aten.silu_, get_raw=True)
def silu_flop(*args, out_val=None, **kwargs):
    if out_val is None:
        raise ValueError("Failed to resolve out_val for SiLU.")

    return out_val.numel() * 5


@register_flop_formula(torch.ops.aten.add, get_raw=True)
@register_flop_formula(torch.ops.aten.add_, get_raw=True)
@register_flop_formula(torch.ops.aten.mul, get_raw=True)
@register_flop_formula(torch.ops.aten.mul_, get_raw=True)
def elementwise_flop(*args, out_val=None, **kwargs):
    if out_val is None:
        raise ValueError("Failed to resolve out_val for element-wise op.")

    return out_val.numel() * 1


@register_flop_formula(torch.ops.aten.rms_norm, get_raw=True)
@register_flop_formula(torch.ops.aten.native_layer_norm, get_raw=True)
def norm_flop(*args, out_val=None, **kwargs):
    if out_val is None:
        raise ValueError("Failed to resolve out_val for norm op.")

    # Norm ops return a tuple: (output, mean, rstd). We only want the output tensor.
    out_tensor = out_val[0] if isinstance(out_val, (tuple, list)) else out_val

    return out_tensor.numel() * 5
