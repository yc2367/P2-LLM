import torch
import numpy as np
import time, argparse, os
from typing import Optional, Dict, List
import pickle


def calc_mem(model_config: Dict, layer_config: Dict, context_length: int=512, batch_size: int=1):
    layer_names         = layer_config.keys()
    num_hidden_layers   = model_config['num_hidden_layers']
    hidden_size         = model_config['hidden_size']
    intermediate_size   = model_config['intermediate_size']
    num_attention_heads = model_config['num_attention_heads']
    if 'num_key_value_heads' in model_config.keys():
        num_key_value_heads = model_config['num_key_value_heads']
    else:
        num_key_value_heads = num_attention_heads

    weight_mem = 0
    act_mem    = 0
    kv_mem     = 0
    score_mem  = 0

    bytes_per_word = 2

    ################ Prefill Stage ################
    # weight memory
    for layer_name in layer_names:
        weight_shape = layer_config[layer_name]
        weight_mem += (np.prod(weight_shape).item() * bytes_per_word)
    
    # activation memory
    act_mem = batch_size * context_length * (intermediate_size*2 + hidden_size*2) * bytes_per_word
    
    # KV-cache memory
    kv_mem = batch_size * context_length * (hidden_size / num_attention_heads * num_key_value_heads) * 2 * num_hidden_layers * bytes_per_word
    
    # attention score memory
    score_mem = batch_size * context_length**2 * num_attention_heads * bytes_per_word
    
    return weight_mem, act_mem, kv_mem, score_mem


if __name__ == '__main__':
    model_name_dict = {
        "meta-llama/Llama-2-7b-hf": "llama_2_7b", 
        # "meta-llama/Llama-2-13b-hf": "llama_2_13b", 
        "meta-llama/Llama-3.1-8B": "llama_3p1_8b", 
        "meta-llama/Llama-3.2-3B": "llama_3p2_3b", 
        "mistralai/Mistral-7B-v0.3": "mistral_7b", 
    }

    base_path = './model_shape_config'
    model_list = []
    mem_list = []

    for model_key, model_value in model_name_dict.items():
        file_path = f'{base_path}/{model_value}.pickle'
        with open(file_path, 'rb') as f:
            model_config, layer_config = pickle.load(f)
        
        weight_mem, act_mem, kv_mem, score_mem = calc_mem(model_config, layer_config, context_length=2048, batch_size=16)

        weight_mem = weight_mem / 1024**3
        act_mem    = act_mem / 1024**3
        kv_mem     = kv_mem / 1024**3
        score_mem  = score_mem / 1024**3

        print(model_key)
        print(f'Weight memory:      {weight_mem} GB')
        print(f'Activation memory:  {act_mem} GB')
        print(f'KV Cache memory:    {kv_mem} GB')
        print(f'Attn Score memory:  {score_mem} GB')
        print('\n')

        model_list.append(model_value)
        mem_list.append((weight_mem, act_mem, kv_mem, score_mem))
    
    print(model_list)
    print(mem_list)
    
    
