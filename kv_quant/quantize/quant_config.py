
from typing import Optional

class QuantConfig(dict):
    def __init__(
        self,
        # general quantization parameters
        w_bits: Optional[int]=None,
        a_bits: Optional[int]=None,
        q_bits: Optional[int]=None,
        k_bits: Optional[int]=None,
        v_bits: Optional[int]=None,
        p_bits: Optional[int]=None,
        w_group_size: Optional[int]=None,
        a_group_size: Optional[int]=None,
        q_group_size: Optional[int]=None,
        k_group_size: Optional[int]=None,
        v_group_size: Optional[int]=None,
        # KV-cache quantization config
        kv_quant_method: Optional[str]=None,
        kv_residual_len: int=0,
        kv_quant_before_attn: bool=False, # If True, KV-cache will be quantized before self-attention
        apply_k_bias: bool=False,
        apply_k_scale: bool=False,
    ):
        for nbits in [w_bits, k_bits, v_bits]:
            assert (nbits is None) or (nbits <= 0) or (nbits in [4, 6, 8, 16]), \
                f'Invalid precision \"{nbits}\" provided for weight / KV-cache. Allowed precisions are {{4, 6, 8, 16}}'
        for nbits in [a_bits, q_bits]:
            assert (nbits is None) or (nbits <= 0) or (nbits in [8, 16]), \
                f'Invalid precision \"{nbits}\" provided for activation / query. Allowed precisions are {{8, 16}}'
        for nbits in [p_bits]:
            assert (nbits is None) or (nbits <= 0) or (nbits in [8, 12, 16]), \
                f'Invalid precision \"{nbits}\" provided for attention-score. Allowed precisions are {{8, 12, 16}}'
        
        for group_size in [w_group_size, a_group_size, q_group_size, k_group_size, v_group_size]:
            assert (group_size <= 0) or (group_size is None) or (group_size in [32, 64, 128]), \
                f'Invalid precision \"{nbits}\" provided for activation / query. Allowed precisions are {{8, 16}}'
        
        if kv_quant_method == 'KCVT':
            assert kv_residual_len % k_group_size == 0, \
                f"The KV residual length {self.kv_residual_len} should be a multiple of the key group size {k_group_size}"
            # assert (kv_residual_len in [64, 128]), \
            #     f'Invalid KV residual length \"{kv_residual_len}\" provided for KCVT quantization. Allowed values are {{64, 128}}'

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.q_bits = q_bits
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.p_bits = p_bits
        self.w_group_size = w_group_size
        self.a_group_size = a_group_size
        self.q_group_size = q_group_size
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size
        self.p_group_size = None # don't apply group-wise quantization for attention-score

        # KV-cache quantization config
        self.kv_quant_method = kv_quant_method
        self.kv_quant_before_attn = kv_quant_before_attn
        self.kv_residual_len = kv_residual_len
        self.apply_k_bias = apply_k_bias
        self.apply_k_scale = apply_k_scale
