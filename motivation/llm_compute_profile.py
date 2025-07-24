import torch
import numpy as np
import time, argparse, os
from typing import Optional, Dict, List
import pickle


def calc_comp(model_config: Dict, layer_config: Dict, context_length: int=512, batch_size: int=1):
    layer_names         = layer_config.keys()
    num_hidden_layers   = model_config['num_hidden_layers']
    hidden_size         = model_config['hidden_size']
    intermediate_size   = model_config['intermediate_size']
    num_attention_heads = model_config['num_attention_heads']
    if 'num_key_value_heads' in model_config.keys():
        num_key_value_heads = model_config['num_key_value_heads']
    else:
        num_key_value_heads = num_attention_heads

    weight_comp = 0
    act_comp    = 0
    kv_comp     = 0
    score_comp  = 0

    op_per_mac  = 2

    ################ Prefill Stage ################
    # weight compute
    for layer_name in layer_names:
        weight_shape = layer_config[layer_name]
        weight_comp += (np.prod(weight_shape).item() * 1 * batch_size * op_per_mac)
    
    # activation compute
    act_comp = weight_comp + (batch_size * context_length * hidden_size * num_hidden_layers * op_per_mac)
    
    # KV-cache compute
    kv_comp = batch_size * context_length * hidden_size * 2 * num_hidden_layers * op_per_mac
    
    # attention score compute
    score_comp = batch_size * context_length * hidden_size * num_hidden_layers * op_per_mac
    
    return weight_comp, act_comp, kv_comp, score_comp


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
    comp_list = []

    for model_key, model_value in model_name_dict.items():
        file_path = f'{base_path}/{model_value}.pickle'
        with open(file_path, 'rb') as f:
            model_config, layer_config = pickle.load(f)
        
        weight_comp, act_comp, kv_comp, score_comp = calc_comp(model_config, layer_config, context_length=2048, batch_size=16)

        weight_comp = weight_comp / 1024**3
        act_comp    = act_comp / 1024**3
        kv_comp     = kv_comp / 1024**3
        score_comp  = score_comp / 1024**3

        print(model_key)
        print(f'Weight compute:      {weight_comp} GOPS')
        print(f'Activation compute:  {act_comp} GOPS')
        print(f'KV Cache compute:    {kv_comp} GOPS')
        print(f'Attn Score compute:  {score_comp} GOPS')
        print('\n')

        model_list.append(model_value)
        comp_list.append((act_comp, score_comp, kv_comp, weight_comp))
    
    print(model_list)
    print(comp_list)
    
    
