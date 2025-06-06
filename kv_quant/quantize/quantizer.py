import torch
from typing import Optional


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


@torch.no_grad()
def k_quant_per_token(
    x_fp: torch.Tensor, q_bits: int=4, group_size: int=128,
    apply_k_bias: Optional[bool]=False, k_bias: Optional[torch.Tensor]=None,
    apply_k_scale: Optional[bool]=False, k_scale: Optional[torch.Tensor]=None,
):
    """
    Asymmetric per-token key quantization.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    :param apply_k_bias: whether to apply per-channel bias subtraction
    :param k_bias: if apply_k_bias == True, then the input tensor's channel subtract k_bias
    :param apply_k_scale: whether to apply per-channel scaling
    :param k_scale: if apply_k_scale == True, then the input tensor's channel will be smoothed (divided) by k_scale
    """
    if q_bits == 16:
        return x_fp

    if apply_k_bias and (not apply_k_scale):
        x_fp_new = x_fp.to(torch.float32) - k_bias.to(torch.float32)
    elif apply_k_scale and (not apply_k_bias):
        x_fp_new = x_fp / k_scale
    elif apply_k_scale and apply_k_bias:
        x_fp_new = (x_fp - k_bias) / k_scale
    else:
        x_fp_new = x_fp
    
    # print(x_fp_new.max(), x_fp_new.min())
    # print(k_bias)
    # print('\n')
    #################### Draw Key Cache and observe ####################
    # X = np.arange(0, x_fp_new[0, 0].shape[1]) 
    # Y = np.arange(0, x_fp_new[0, 0].shape[0])
    # X, Y = np.meshgrid(X, Y)

    # Z = x_fp[0, 0].clone().cpu().abs() 
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, rstride=10, cstride=1, cmap='coolwarm', linewidth=0.05, antialiased=True)
    # fig.colorbar(surf)
    # fig.savefig('/home/yc2367/llm/P2-LLM/kv_quant/full.png', bbox_inches = 'tight', format='png', dpi=200, pad_inches=0.1)

    # Z = x_fp_new[0, 0].clone().cpu().abs() 
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, rstride=10, cstride=1, cmap='coolwarm', linewidth=0.05, antialiased=True)
    # fig.colorbar(surf)
    # fig.savefig('/home/yc2367/llm/P2-LLM/kv_quant/quant.png', bbox_inches = 'tight', format='png', dpi=200, pad_inches=0.1)
    
    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp_new = (
        x_fp_new.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    ).to(torch.float16)

    num_groups = (num_head * h_dim) // group_size
    assert num_groups * group_size == num_head * h_dim, \
        f"The input tensor's last dimension {x_fp_new.shape[-1]} is not divisible by group_size {group_size}"
    x_fp_new = x_fp_new.view(batch, seq_len, num_groups, group_size)

    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp_new, dim=-1, keepdim=True)
    rmax = torch.amax(x_fp_new, dim=-1, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = (torch.round(-rmin / scale_fp)).clamp_(qmin, qmax)
    x_q  = torch.clamp(torch.round(x_fp_new / scale_fp) + zeropoint, min=qmin, max=qmax)
    x_dq = (x_q - zeropoint) * scale_fp # de-quantized tensor

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3)
    if apply_k_scale:
        x_dq = x_dq * k_scale
    if apply_k_bias:
        x_dq = x_dq.to(torch.float32) + k_bias.to(torch.float32)
    
    #print(f'Quant Error: {(x_dq - x_fp).pow(2).mean()}')
    return x_dq.to(torch.float16)


@torch.no_grad()
def k_quant_per_channel(
    x_fp: torch.Tensor, q_bits: int=4, group_size: int=128,
):
    """
    Asymmetric per-channel key quantization for KIVI.
    NOTE: The zero-point is in FP16.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    """
    if q_bits >= 16:
        return x_fp

    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp_new = (
        x_fp.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )
    #NOTE(Trial): See the influence of fp16
    #.float()
    num_groups = seq_len // group_size
    assert num_groups * group_size == seq_len, \
        f"The input tensor's sequence length {seq_len} is not divisible by group_size {group_size}"
    x_fp_new = x_fp_new.view(batch, num_groups, group_size, num_head * h_dim)
    
    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp_new, dim=-2, keepdim=True)
    rmax = torch.amax(x_fp_new, dim=-2, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = -rmin
    q_tensor = torch.clamp(torch.round((x_fp_new + zeropoint) / scale_fp), min=qmin, max=qmax)
    x_dq = (q_tensor * scale_fp) - zeropoint

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3)
    #print(f'Quant Error: {(x_dq - x_fp).pow(2).mean()}')
    return x_dq


@torch.no_grad()
def v_quant_per_token(
    x_fp: torch.Tensor, q_bits: int=4, group_size: int=128
):
    """
    Asymmetric per-token value quantization.

    :param x_fp: input tensor to be quantized
    :param q_bits: quantization bit-width
    :param group_size: quantization group size
    """
    if q_bits >= 16:
        return x_fp

    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp_new = (
        x_fp.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )
    #NOTE(Trial): See the influence of fp16
    #.float()

    num_groups = (num_head * h_dim) // group_size
    assert num_groups * group_size == num_head * h_dim, \
        f"The input tensor's last dimension {x_fp_new.shape[-1]} is not divisible by group_size {group_size}"
    x_fp_new = x_fp_new.view(batch, seq_len, num_groups, group_size)

    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp_new, dim=-1, keepdim=True)
    rmax = torch.amax(x_fp_new, dim=-1, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = (torch.round(-rmin / scale_fp)).clamp_(qmin, qmax)
    x_q  = torch.clamp(torch.round(x_fp_new / scale_fp) + zeropoint, min=qmin, max=qmax)
    x_dq = (x_q - zeropoint) * scale_fp # de-quantized tensor

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3)
    return x_dq


def kv_quant_function(
    k_fp, 
    v_fp, 
    quant_config, 
    k_bias: Optional[torch.Tensor]=None,
    k_scale: Optional[torch.Tensor]=None,
):
    kv_quant_method = quant_config.kv_quant_method
    k_bits = quant_config.k_bits
    v_bits = quant_config.v_bits
    k_group_size = quant_config.k_group_size
    v_group_size = quant_config.v_group_size
    assert kv_quant_method in ['KTVT', 'KCVT'], \
        f'Invalid quantization method \"{kv_quant_method}\" provided. ' + \
        'Currently only support \"KTVT\" and \"KCVT\" quantization.'

    if kv_quant_method == "KTVT": # key per-token, value per-token
        k_dq = k_quant_per_token(
            k_fp, k_bits, k_group_size,
            apply_k_bias=quant_config.apply_k_bias, k_bias=k_bias,
            apply_k_scale=quant_config.apply_k_scale, k_scale=k_scale
        )
        v_dq = v_quant_per_token(
            v_fp, v_bits, v_group_size
        )
    elif kv_quant_method == "KCVT": # key per-channel, value per-token
        k_dq = k_quant_per_channel(
            k_fp, k_bits, k_group_size
        )
        v_dq = v_quant_per_token(
            v_fp, v_bits, v_group_size
        )        

    return k_dq, v_dq
