from typing import List
import numpy as np
import pickle

from utils import MODEL_NAME_DICT


class PCU_Array:
    PR_SCALING = 1.25 # scaling factor to account for post placement and routing

    ## The class constructor
    # @param model_name:       Name of the model to be evaluated.
    # @param batch_size:       Batch size.
    # @param cxt_len:          Iput context length.

    # @param w_prec:           Weight precision.
    # @param a_prec:           Input activation precision.
    # @param q_prec:           Query precision.
    # @param p_prec:           Attention-score precision.
    # @param kv_prec:          KV-cache precision.

    # @param page_size:        DRAM page size (i.e., number of bits in a row).
    # @param num_channel:      Number of DRAM channels.
    # @param pcu_per_channel:  Number of PIM Compute Units (PCUs) per channel.
    # @param pe_per_pcu:       Number of PEs per PCU.
    # @param pe_dp_size:       Dot-product size of the PE.
    # @param pcu_reuse_factor: Temporal reuse factor for PCU within tCCD_L.
    # @param pcu_energy:       Energy cost per PCU.    

    # @param tRCD:             Row-to-Column Delay. 
    # @param tRP:              Row Precharge Delay.    
    # @param tCCD_L:           Column-to-Column Delay, same back group.   
    # @param tCCD_S:           Column-to-Column Delay, different back groups. 
          
    def __init__(
        self,
        model_name: str,
        batch_size: int=1,
        cxt_len: int=4096,

        w_prec: int=16, 
        a_prec: int=16, 
        q_prec: int=16,
        p_prec: int=16,
        kv_prec: int=16,

        page_size: int=8192,
        num_channel: int=16,
        pcu_per_channel: int=8,
        pe_per_pcu: int=16,
        pe_dp_size: int=1,
        pcu_reuse_factor: int=1,
        pcu_energy: float=0, 

        tRCD: int=14,
        tRP: int=14,
        tCCD_L: int=4,
        tCCD_S: int=2,
    ):
        assert pcu_energy >= 0, "ERROR! You must provide the energy cost of a PCU."
        
        self.model_name       = model_name
        self.batch_size       = batch_size
        self.cxt_len          = cxt_len

        self.w_prec           = w_prec
        self.a_prec           = a_prec
        self.q_prec           = q_prec
        self.p_prec           = p_prec
        self.kv_prec          = kv_prec

        self.page_size        = page_size
        self.num_channel      = num_channel
        self.pcu_per_channel  = pcu_per_channel
        self.pe_per_pcu       = pe_per_pcu
        self.pe_dp_size       = pe_dp_size
        self.pcu_reuse_factor = pcu_reuse_factor
        self.pcu_energy       = pcu_energy * self.PR_SCALING

        self.tRCD             = tRCD
        self.tRP              = tRP
        self.tCCD_L           = tCCD_L
        self.tCCD_S           = tCCD_S

        self.pe_per_channel   = pcu_per_channel * pe_per_pcu
        self.pe_num_total     = num_channel * pcu_per_channel * pe_per_pcu
        self._init_model_profiler(model_name, cxt_len)
    
    def _init_model_profiler(self, model_name, cxt_len: int=4096):
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
        attn_group_size = num_attention_heads / num_key_value_heads

        for l_idx in range(num_hidden_layers):
            op_name = f'model.layers.{l_idx}.self_attn.attn_qk'
            weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size] # key dimension
            input_dim[op_name]  = [batch_size * num_key_value_heads, attn_group_size, head_size] # query dimension
            output_dim[op_name] = [batch_size * num_key_value_heads, attn_group_size, cxt_len] # score dimension
            
            op_name = f'model.layers.{l_idx}.self_attn.attn_pv'
            weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len] # value dimension
            input_dim[op_name]  = [batch_size * num_key_value_heads, attn_group_size, cxt_len] # score dimension
            output_dim[op_name] = [batch_size * num_key_value_heads, attn_group_size, head_size] # output dimension

        self.weight_dim = weight_dim
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.layer_name_list = list(weight_dim.keys())

    def _init_mem(self):
        raise NotImplementedError('ERROR! No implementation of function _init_mem()')

