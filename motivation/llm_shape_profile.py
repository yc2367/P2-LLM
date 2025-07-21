import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import argparse, os
from typing import Optional
import pickle

torch.set_grad_enabled(False)


if __name__ == '__main__':
    model_name_dict = {
        "meta-llama/Llama-2-7b-hf": "llama_2_7b", 
        "meta-llama/Llama-2-13b-hf": "llama_2_13b", 
        "meta-llama/Llama-3.1-8B": "llama_3p1_8b", 
        "meta-llama/Llama-3.2-3B": "llama_3p2_3b", 
        "mistralai/Mistral-7B-v0.3": "mistral_7b", 
    }

    for model_key in model_name_dict.keys():
        model = AutoModelForCausalLM.from_pretrained(model_key, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
        model_config = AutoConfig.from_pretrained(model_str).to_dict()

        layer_config = {}
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                layer_config[n] = list(m.weight.shape)
                print(f'Module name:  {n}')
                print(f'Module shape: {m.weight.shape}')
                print()
        print('\n\n')

        file_path = f'./model_shape_config/{model_name_dict[model_key]}.pickle'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump((model_config, layer_config), f)
