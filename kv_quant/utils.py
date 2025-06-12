import argparse
import importlib
import numpy as np
import random, torch
from functools import reduce
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaConfig
from models import QuantLlamaForCausalLM
from quantize import QuantConfig

import json
import os
model2path = json.load(open(os.path.join(os.path.dirname(__file__), "model2path.json"), "r"))


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str, help="model to load")
    parser.add_argument('--use_fp16', action="store_true", default=False, help="Whether to use the original FP16 model.")
    return parser


def add_quant_args(parser):
    parser.add_argument('--w_bits', type=int, default=16, help="Number of bits for weight quantization.")
    parser.add_argument('--a_bits', type=int, default=16, help="Number of bits for activation quantization.")
    parser.add_argument('--q_bits', type=int, default=16, help="Number of bits for query quantization.")
    parser.add_argument('--k_bits', type=int, default=16, help="Number of bits for key quantization.")
    parser.add_argument('--v_bits', type=int, default=16, help="Number of bits for value quantization.")
    parser.add_argument('--p_bits_pf', type=int, default=16, help="Number of bits for attention-score quantization during prefill.")
    parser.add_argument('--p_bits_dc', type=int, default=16, help="Number of bits for attention-score quantization during decode.")
    parser.add_argument('--w_group_size', type=int, default=-1, help="Group size for weight quantization.")
    parser.add_argument('--a_group_size', type=int, default=-1, help="Group size for activation quantization.")
    parser.add_argument('--q_group_size', type=int, default=-1, help="Group size for query quantization.")
    parser.add_argument('--k_group_size', type=int, default=-1, help="Group size for key quantization.")
    parser.add_argument('--v_group_size', type=int, default=-1, help="Group size for value quantization.")
    parser.add_argument("--kv_quant_method", type=str, default="KCVT", help="KV-cache quantization method: KCVT / KTVT.")
    parser.add_argument("--kv_residual_len", type=int, default=1, help="Residual length (number of tokens maintained in FP16) for KV-cache quantization.")
    parser.add_argument("--kv_quant_post_attn", action="store_true", default=False, help="Whether to apply KV-cache quantization before or after self-attention.")
    parser.add_argument("--apply_k_bias", action="store_true", default=False, help="Whether to apply per-channel key scaling for KTVT quantization")
    parser.add_argument("--apply_k_scale", action="store_true", default=False, help="Whether to apply per-channel key bias subtraction for KTVT quantizationT")

    return parser
    

def get_quant_config(args):
    quant_config = QuantConfig(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        q_bits=args.q_bits,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        p_bits_pf=args.p_bits_pf,
        p_bits_dc=args.p_bits_dc,
        w_group_size=args.w_group_size,
        a_group_size=args.a_group_size,
        q_group_size=args.q_group_size,
        k_group_size=args.k_group_size,
        v_group_size=args.v_group_size,
        kv_quant_method=args.kv_quant_method,
        kv_residual_len=args.kv_residual_len,
        kv_quant_post_attn=args.kv_quant_post_attn,
        apply_k_bias=args.apply_k_bias,
        apply_k_scale=args.apply_k_scale,
    )
    return quant_config


# Set seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3
    return size_all_mb


def get_module_by_name(module, module_name):
    names = module_name.split(sep='.')
    return reduce(getattr, names, module)


def load_model_and_tokenizer(model_name_or_path, quant_config=None, device_map="cuda", use_fp16: bool=False, use_slow_attn: bool=False):
    """
    Args:
        model_name_or_path: The model to be evaluated.
        quant_config: The quantization configuration. Will be discarded if "use_fp16=True".
        device_map: "cpu" or "cuda".
        use_fp16: If set to True, then evaluate the original FP16 model.
        use_slow_attn: If set to True, then use a for loop to iterate over the number of heads during self-attention to avoid OOM for LongBench dataset.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if 'llama' in model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_name_or_path)
        if use_fp16:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map
            )
        else:
            config.use_slow_attn = use_slow_attn
            model = QuantLlamaForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
                quant_config=quant_config
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    
    model.eval()        
    return model, tokenizer