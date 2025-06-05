import argparse
import importlib
import numpy as np
import random, torch
from functools import reduce
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache
from loguru import logger

from BitMod import CompressionConfig, convert_attention_to_experimental

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# Set seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model_numel(model):
    param_cnt = 0
    for name, module in model.named_modules():
        if hasattr(module, '_nelement'):
            param_cnt += module._nelement()
    return param_cnt

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

def get_compression_config(args):
    config = CompressionConfig(
        compress_method=args.compress_method,
        rank=args.rank,
        rankv=args.rankv,
        prefill_rank = args.prefillrank,
        prefill_rankv = args.prefillrankv,
        loop=args.loop,
        quantize_bit=args.quantize_bit,
        group_num=args.group_num,
        group_size = args.group_size,
        top_k=args.top_kprun,
        left=args.left,
        attention_number=args.attention_number,
        device_num=args.gpu,
        batch_num=args.batch_size,
        streaming=args.streaming,
        streaming_gap=args.streaming_gap,
        stream_grouping=args.stream_grouping,
        use_bitmod=args.use_bitmod,
    )
    return config


def load_model_and_tokenizer(model_name_or_path, use_flash_attn2=False, device_map="cuda", compression_config=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attn2 else "sdpa",
    )
    
    if compression_config is not None:
        logger.info("Found compression configs...")
        logger.info("Converting attention to experimental...")
        convert_attention_to_experimental(model, compression_config)
    model.eval()        
    return model, tokenizer



def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name_or_path', type=str, help='model to load')
    return parser

def add_compression_args(parser):
    parser.add_argument("--compress_method", type=str, default=None, help="")
    parser.add_argument("--rank", type=float, default=0.0, help="")
    parser.add_argument("--rankv", type=float, default=0.0, help="")
    parser.add_argument("--loop", type=int, default=0, help="")
    parser.add_argument("--quantize_bit", type=int, default=8, help="")
    parser.add_argument("--group_num", type=int, default=0, help="")
    parser.add_argument("--group_size", type=int, default=0, help="")
    parser.add_argument("--top_kprun", type=float, default=0.0, help="")
    parser.add_argument("--left", type=float, default=0.0, help="")
    parser.add_argument("--attention_number", type=int, default=100, help="")
    parser.add_argument("--gpu", type=int, default=0, help="")
    return parser


def add_cache_args(parser):
    parser.add_argument('--nbits', type=int, default=16, help='Number of bits for quantization. Use 16 for default DynamicCache')

    # Separate quantization parameters for keys and values
    parser.add_argument('--method_key', type=str, default='classic', help='Quantization method for keys')
    parser.add_argument('--method_value', type=str, default='classic', help='Quantization method for values')
    parser.add_argument('--clip_ratio_key', type=float, default=1.0, help='Clip ratio for quantization of keys')
    parser.add_argument('--clip_ratio_value', type=float, default=1.0, help='Clip ratio for quantization of values')
    parser.add_argument('--datatype_key', type=str, default='mixed', help='Datatype for quantization of keys')
    parser.add_argument('--datatype_value', type=str, default='mixed', help='Datatype for quantization of values')

    parser.add_argument('--sym', action='store_true', help='Use symmetric quantization')
    # Other cache parameters
    parser.add_argument('--axis_key', type=int, default=-1, help='Axis for key quantization')
    parser.add_argument('--axis_value', type=int, default=-1, help='Axis for value quantization')
    parser.add_argument('--q_group_size', type=int, default=128, help='Group size for quantization')
    parser.add_argument('--residual_length', type=int, default=128, help='Residual length for cache')
    parser.add_argument('--prefill_quant', action='store_true', help='Quantize during prefill phase')
