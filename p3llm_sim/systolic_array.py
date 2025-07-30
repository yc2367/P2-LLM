from typing import List
import numpy as np
import pickle

from utils import MODEL_NAME_DICT


class SystolicArray:
    PR_SCALING = 1.25 # scaling factor to account for post placement and routing

    ## The class constructor
    # @param model_name:    Name of the model to be evaluated.
    # @param batch_size:    Batch size.

    # @param w_prec:        Weight precision.
    # @param a_prec:        Input activation precision.
    # @param q_prec:        Query precision.
    # @param p_prec:        Attention-score precision.
    # @param kv_prec:       KV-cache precision.

    # @param num_npu_core:  Number of NPU cores.
    # @param pe_dp_size:    Dot-product size of the PE.
    # @param pe_energy:     Energy cost of PE.
    # @param cxt_len:       Iput context length.
    # @param is_prefill:    Whether the simulation mode is the prefill stage.
    def __init__(
        self,
        model_name: str,
        batch_size: int=1,
        w_prec: int=16, 
        a_prec: int=16, 
        q_prec: int=16,
        p_prec: int=16,
        kv_prec: int=16,
        num_npu_core: int=4,
        pe_dp_size: int=1,
        pe_energy: float=0, 
        pe_array_dim: List[int]=[],
        cxt_len: int=4096,
        is_prefill: bool=False,
    ):
        assert pe_energy >= 0, "ERROR! You must provide the energy cost of a PE."
        assert len(pe_array_dim) == 2, f"ERROR! The dimension of PE array must be 2. But you gave {len(pe_array_dim)}."
        
        self.model_name    = model_name
        self.batch_size    = batch_size
        self.cxt_len       = cxt_len

        self.w_prec        = w_prec
        self.a_prec        = a_prec
        self.q_prec        = q_prec
        self.p_prec        = p_prec
        self.kv_prec       = kv_prec

        self.num_npu_core  = num_npu_core
        self.pe_dp_size    = pe_dp_size
        self.total_pe_num  = np.prod(pe_array_dim)
        self.pe_energy     = pe_energy * self.PR_SCALING
        self.pe_array_dim  = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        
        self._init_model_profiler(model_name, cxt_len, is_prefill)
    
    def _init_model_profiler(self, model_name, cxt_len: int=2048, is_prefill: bool=False):
        file_path = f'./model_shape_config/{MODEL_NAME_DICT[model_name]}.pickle'
        with open(file_path, 'rb') as f:
            model_config, layer_config = pickle.load(f)
        
        ########## FFN Dimension ##########
        batch_size = self.batch_size
        weight_dim = {}
        input_dim  = {}
        output_dim = {}
        for name, weight_shape in layer_config.items():
            weight_dim[name] = [1] + weight_shape
            if is_prefill: # prefill
                input_dim[name]  = [batch_size, cxt_len, weight_shape[1]]
                output_dim[name] = [batch_size, cxt_len, weight_shape[0]]
            else:
                input_dim[name]  = [1, batch_size, weight_shape[1]]
                output_dim[name] = [1, batch_size, weight_shape[0]]

        ########## Attention Dimension ##########
        num_hidden_layers   = model_config['num_hidden_layers']
        hidden_size         = model_config['hidden_size']
        num_attention_heads = model_config['num_attention_heads']
        head_size           = hidden_size / num_attention_heads
        if 'num_key_value_heads' in model_config.keys():
            num_key_value_heads = model_config['num_key_value_heads']
        else:
            num_key_value_heads = num_attention_heads
        query_share_factor = num_attention_heads / num_key_value_heads

        for l_idx in range(num_hidden_layers):
            op_name = f'model.layers.{l_idx}.self_attn.attn_qk'
            if is_prefill: # generation
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size] # key dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size] # query dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len] # score dimension
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size] # key dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor, head_size] # query dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, cxt_len] # score dimension
            
            op_name = f'model.layers.{l_idx}.self_attn.attn_pv'
            if is_prefill: # generation
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len] # value dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len] # score dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size] # output dimension
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len] # value dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor, cxt_len] # score dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, head_size] # output dimension

        self.weight_dim = weight_dim
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.layer_name_list = list(weight_dim.keys())

    def _init_mem(self):
        raise NotImplementedError('ERROR! No implementation of function _init_mem()')

