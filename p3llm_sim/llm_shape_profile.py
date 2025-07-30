import torch
torch.set_grad_enabled(False)

from transformers import AutoModelForCausalLM, AutoConfig
import argparse, os
import pickle

from utils import MODEL_NAME_DICT



if __name__ == '__main__':
    for model_key in MODEL_NAME_DICT.keys():
        model = AutoModelForCausalLM.from_pretrained(model_key, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
        model_config = AutoConfig.from_pretrained(model_key).to_dict()

        layer_config = {}
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                layer_config[n] = list(m.weight.shape)
                print(f'Module name:  {n}')
                print(f'Module shape: {m.weight.shape}')
                print()
        print('\n\n')

        file_path = f'./model_shape_config/{MODEL_NAME_DICT[model_key]}.pickle'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump((model_config, layer_config), f)
