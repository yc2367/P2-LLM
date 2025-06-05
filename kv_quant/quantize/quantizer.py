import torch
from typing import Optional


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

    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp = (
        x_fp.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )
    #NOTE(Trial): See the influence of fp16
    #.float()
    if apply_k_bias:
        x_fp = x_fp - k_bias
    if apply_k_scale:
        x_fp = x_fp / k_scale

    num_groups = (num_head * h_dim) // group_size
    assert num_groups * group_size == num_head * h_dim, \
        f"The input tensor's last dimension {x_fp.shape[-1]} is not divisible by group_size {group_size}"
    x_fp = x_fp.view(batch, seq_len, num_groups, group_size)

    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp, dim=-1, keepdim=True)
    rmax = torch.amax(x_fp, dim=-1, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = (torch.round(-rmin / scale_fp)).clamp_(qmin, qmax)
    x_q  = torch.clamp(torch.round(x_fp / scale_fp) + zeropoint, min=qmin, max=qmax)
    x_dq = (x_q - zeropoint) * scale_fp # de-quantized tensor

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3)
    return x_dq


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
    if q_bits == 16:
        return x_fp

    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp = (
        x_fp.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )
    #NOTE(Trial): See the influence of fp16
    #.float()
    num_groups = seq_len // group_size
    assert num_groups * group_size == seq_len, \
        f"The input tensor's sequence length {seq_len} is not divisible by group_size {group_size}"
    x_fp = x_fp.view(batch, num_group, group_size, num_head * h_dim)
    
    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp, dim=-2, keepdim=True)
    rmax = torch.amax(x_fp, dim=-2, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = -rmin
    q_tensor = torch.clamp(torch.round((x_fp + zeropoint) / scale_fp), min=qmin, max=qmax)
    x_dq = (q_tensor * scale_fp) - zeropoint

    x_dq = x_dq.view(batch, seq_len, num_head, h_dim).permute(0, 2, 1, 3)
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
    if q_bits == 16:
        return x_fp

    batch, num_head, seq_len, h_dim = x_fp.shape
    x_fp = (
        x_fp.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, num_head * h_dim)
    )
    #NOTE(Trial): See the influence of fp16
    #.float()

    num_groups = (num_head * h_dim) // group_size
    assert num_groups * group_size == num_head * h_dim, \
        f"The input tensor's last dimension {x_fp.shape[-1]} is not divisible by group_size {group_size}"
    x_fp = x_fp.view(batch, seq_len, num_groups, group_size)

    qmin = 0
    qmax = 2**q_bits - 1
    rmin = torch.amin(x_fp, dim=-1, keepdim=True)
    rmax = torch.amax(x_fp, dim=-1, keepdim=True)
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e5)
    zeropoint = (torch.round(-rmin / scale_fp)).clamp_(qmin, qmax)
    x_q  = torch.clamp(torch.round(x_fp / scale_fp) + zeropoint, min=qmin, max=qmax)
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

    batch, num_head, seq_len, h_dim = k_fp.shape

    if kv_quant_method == "KTVT": # key per-token, value per-token
        k_dq = k_quant_per_token(
            k_fp, k_bits, k_group_size,
            quant_config.apply_k_bias, k_bias,
            quant_config.apply_k_scale, k_scale
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




def compress_insert_function_old(
    k_fp,
    v_fp,
    quant_config,
    layer_idx,
):
    batch, num_head, seq_len, h_dim = k_fp.shape
    if quant_config.token_preserving[layer_idx] == True:
        starting_idx = int(quant_config.start_saving[layer_idx] * seq_len)
        locality_idx = int(quant_config.locality_saving[layer_idx] * seq_len)
    else:
        starting_idx = int(0)
        locality_idx = -seq_len
        
    if quant_config.method[layer_idx] == "PTG": #per-token group-wise
        if quant_config.use_bitmod[layer_idx] == True:
            k_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization_bitmod(
                k_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                h_dim,
                datatype='mixed',
            )
            if v_fp is not None:
                v_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization_bitmod(
                    v_fp[:, :, starting_idx:-locality_idx, :],
                    quant_config.quantize_bit[layer_idx],
                    h_dim,
                    datatype='mixed',
                )
        else:
            k_fp[:, :, starting_idx:-locality_idx, :] =  fake_groupwise_token_asymmetric_quantization(
                k_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                h_dim
            )
            if v_fp is not None:
                v_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                    v_fp[:, :, starting_idx:-locality_idx, :],
                    quant_config.quantize_bit[layer_idx],
                    h_dim
                )
    if quant_config.method[layer_idx] == "KCVT":
        if quant_config.use_bitmod[layer_idx] == True:
            k_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_bitmod(
                k_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                seq_len,
                datatype='mixed',
            )
            if v_fp is not None:
                v_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization_bitmod(
                    v_fp[:, :, starting_idx:-locality_idx, :],
                    quant_config.quantize_bit[layer_idx],
                    int(num_head * h_dim),
                    datatype='mixed',
                )
        else:
            k_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
                k_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                seq_len,
            )
            if v_fp is not None:
                v_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                    v_fp[:, :, starting_idx:-locality_idx, :],
                    quant_config.quantize_bit[layer_idx],
                    int(num_head * h_dim),
                )

    if quant_config.method[layer_idx] == "KIVI":
        if quant_config.use_bitmod[layer_idx] == True:
            k_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_bitmod(
                k_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                quant_config.group_size[layer_idx],
                'mixed'
            )
            v_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization_bitmod(
                v_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                quant_config.group_size[layer_idx],
                'mixed'
            )
        else:
            k_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
                k_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                quant_config.group_size[layer_idx]
            )
            v_fp[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                v_fp[:, :, starting_idx:-locality_idx, :],
                quant_config.quantize_bit[layer_idx],
                quant_config.group_size[layer_idx]
            )
    return k_fp, v_fp

