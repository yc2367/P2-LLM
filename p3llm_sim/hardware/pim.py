import math
import numpy as np

from mem import MemoryInstance
from .pcu_array import PCU_Array


class PIM(PCU_Array):
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
        init_mem: bool=True,
    ): 
        super().__init__(
            model_name=model_name,
            batch_size=batch_size, 
            cxt_len=cxt_len, 
            w_prec=w_prec, 
            a_prec=a_prec,
            q_prec=q_prec,
            p_prec=p_prec,
            kv_prec=kv_prec, 

            page_size=page_size,
            num_channel=num_channel, 
            pcu_per_channel=pcu_per_channel, 
            pe_per_pcu=pe_per_pcu, 
            pe_dp_size=pe_dp_size, 
            pcu_reuse_factor=pcu_reuse_factor, 
            pcu_energy=pcu_energy, 

            tRCD=tRCD, 
            tRP=tRP, 
            tCCD_L=tCCD_L,
            tCCD_S=tCCD_S
        )

        self.cycle_compute = None
        if init_mem:
            self._init_mem()
            self._check_layer_mem_size()

    def calc_cycle(self):
        self._calc_compute_cycle()
        total_cycle_compute = 0
        for layer_name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute[layer_name]
            total_cycle_compute += cycle_layer_compute

        self.cycle_compute = total_cycle_compute

        total_cycle_compute_linear = 0
        total_cycle_compute_attn = 0
        for layer_name in self.layer_name_list:
            if ('attn_qk' in layer_name) or ('attn_pv' in layer_name):
                total_cycle_compute_attn += self._layer_cycle_compute[layer_name]
            else:
                total_cycle_compute_linear += self._layer_cycle_compute[layer_name]
        #NOTE: Uncomment later.
        # print(f'Linear Compute: {total_cycle_compute_linear}')
        # print(f'Attn Compute:   {total_cycle_compute_attn}')
        # print('\n')
        
        return total_cycle_compute
    
    def _calc_compute_cycle(self):
        self._layer_cycle_compute = {}
        for layer_name in self.layer_name_list:
            w_dim = self.weight_dim[layer_name]
            o_dim = self.output_dim[layer_name]

            if ('attn_qk' not in layer_name) and ('attn_pv' not in layer_name):
                cycle_layer_compute = self._calc_cycle_fc(w_dim, o_dim)
            else:
                cycle_layer_compute = self._calc_cycle_attn(w_dim, o_dim, layer_name)
            
            self._layer_cycle_compute[layer_name] = cycle_layer_compute
            
            # print(layer_name)
            # print(w_dim)
            # print(o_dim)
            # print(cycle_layer_compute)
            # print()
        return

    def _calc_cycle_fc(self, w_dim, o_dim):
        page_size        = self.page_size
        pe_dp_size       = self.pe_dp_size
        pe_num_total     = self.pe_num_total
        pcu_reuse_factor = self.pcu_reuse_factor
        pe_per_channel   = self.pe_per_channel

        num_word_per_page = page_size / self.w_prec
        dp_size_per_page = num_word_per_page / self.pe_per_pcu  # dot-product size in one page

        # output channel, input channel
        _, cout, cin = w_dim
        # num token, output channel
        _, batch_size, _ = o_dim

        # tile_cin:    number of tiles along input channel
        # tile_cout:   number of tiles along output channel
        # tile_batch:  number of tiles along batch size
        tile_cin   = math.ceil(cin / pe_dp_size)
        tile_cout  = math.ceil(cout / pe_num_total)
        tile_batch = math.ceil(batch_size / pcu_reuse_factor)

        ###########  Overhead of row activation-precharge before PIM and output register readout after PIM  ###########
        if (pcu_reuse_factor > 1) and (batch_size > 1):
            # Throughput-enhanced PCU
            num_output_per_pe = 2
        else:
            num_output_per_pe = 1
        num_page_activate             = cin / dp_size_per_page
        overhead_input_write_per_page = dp_size_per_page * self.a_prec / (self.dram.rw_bw / self.num_channel) * num_output_per_pe
        overhead_per_page_activate    = max(self.tRCD + self.tRP, overhead_input_write_per_page)
        overhead_cin                  = overhead_per_page_activate * num_page_activate
        overhead_output_readout       = num_output_per_pe * 32 * pe_per_channel / (self.dram.rw_bw / self.num_channel)
        ##############################################################################################################
        cycle_cin = (tile_cin * self.tCCD_L) + overhead_cin + overhead_output_readout
        cycle_total = cycle_cin * tile_cout * tile_batch

        return cycle_total
    
    def _calc_cycle_attn(self, w_dim, o_dim, layer_name):
        page_size        = self.page_size
        num_channel      = self.num_channel
        pe_per_channel   = self.pe_per_channel
        pe_dp_size       = self.pe_dp_size
        pcu_reuse_factor = self.pcu_reuse_factor

        num_word_per_page = page_size / self.kv_prec
        dp_size_per_page = num_word_per_page / self.pe_per_pcu  # dot-product size in one page

        # output channel, input channel
        _, cout, cin = w_dim
        # num token, output channel
        batch_size, attn_group_size, _ = o_dim

        # tile_cin:         number of tiles along input channel
        # tile_cout:        number of tiles along output channel
        # tile_batch:       number of tiles along batch size
        # tile_attn_group:  number of tiles along attention group size
        # print(layer_name)
        # print(f"cout:              {cout}")
        # print(f"cin:               {cin}")
        # print(f"batch_size:        {batch_size}")
        # print(f"attn_group_size:   {attn_group_size}")
        
        tile_cin = math.ceil(cin / pe_dp_size)
        if ('qk' in layer_name) and (batch_size / num_channel < 1):
            tile_cout  = math.ceil(cout / 4 / pe_per_channel)
            tile_batch = math.ceil(batch_size * 4 / num_channel)
        else:
            tile_cout  = math.ceil(cout / pe_per_channel)
            tile_batch = math.ceil(batch_size / num_channel)
        
        tile_attn_group  = math.ceil(attn_group_size / pcu_reuse_factor)
        # print(f"tile cout:              {tile_cout}")
        # print(f"tile cin:               {tile_cin}")
        # print(f"tile attn_group:        {tile_attn_group}")
        # print(f"tile_batch:             {tile_batch}")
        # print()

        # print(w_dim, o_dim)
        # print(f'batch size: {batch_size}')
        # print(f'cout:       {cout}')
        # print(f'cin:        {cin}')
        # print(f'tile_cin:   {tile_cin}')
        # print(f'tile_cout:  {tile_cout}')
        # print(f'tile_batch: {tile_batch}')
        # print(f'tile_attn_group: {tile_attn_group}')
        # print()
        ###########  Overhead of row activation-precharge before PIM and output register readout after PIM  ###########
        if (pcu_reuse_factor > 1) and (attn_group_size > 1):
            # Throughput-enhanced PCU
            num_output_per_pe = 2
        else:
            num_output_per_pe = 1
        num_page_activate             = cin / dp_size_per_page
        overhead_input_write_per_page = dp_size_per_page * self.a_prec / (self.dram.rw_bw / self.num_channel) * num_output_per_pe
        overhead_per_page_activate    = max(self.tRCD + self.tRP, overhead_input_write_per_page)
        overhead_cin                  = overhead_per_page_activate * num_page_activate
        overhead_output_readout       = num_output_per_pe * 32 * pe_per_channel / (self.dram.rw_bw / self.num_channel)
        ##############################################################################################################
        cycle_cin = (tile_cin * self.tCCD_L) + overhead_cin + overhead_output_readout
        cycle_total = cycle_cin * tile_cout * tile_batch * tile_attn_group

        return cycle_total
    
    def calc_compute_energy(self):
        self._layer_energy_compute = {}
        compute_energy_per_cycle = self.num_channel * self.dram.r_cost 
        total_energy = 0

        if self.cycle_compute is None:
            self.cycle_compute = self.calc_cycle()
        
        for layer_name in self.layer_name_list:
            layer_energy = self._layer_cycle_compute[layer_name] * compute_energy_per_cycle
            if (self.pcu_reuse_factor > 1) and (self.batch_size > 1):
                layer_energy = layer_energy * 1.3
            elif (self.pcu_reuse_factor > 1) and (self.batch_size == 1):
                layer_energy = layer_energy * 1.0
            else:
                layer_energy = layer_energy * 0.9
            
            self._layer_energy_compute[layer_name] = layer_energy
            total_energy += layer_energy
        
        return total_energy 
    
    def calc_output_rd_energy(self):  # energy of streaming PCU outputs to NPU for post-processing
        def _calc_output_rd_energy_fc(layer_name):
            dram_bus_width_per_channel = self.dram.rw_bw / self.num_channel
            num_o  = np.prod(self.output_dim[layer_name]).item()
            o_prec = 32
            
            num_o_rd = math.ceil(num_o * o_prec / dram_bus_width_per_channel) 
            energy_o_rd = num_o_rd * self.dram.r_cost

            return energy_o_rd
        
        self._layer_output_rd_energy = {}
        total_energy = 0
        for layer_name in self.layer_name_list:
            layer_energy = _calc_output_rd_energy_fc(layer_name)
            self._layer_output_rd_energy[layer_name] = layer_energy
            total_energy += layer_energy

        return total_energy
    
    def _check_layer_mem_size(self):
        self._w_mem_byte_required = {}
        self._i_mem_byte_required = {}
        self._o_mem_byte_required = {}   

        for _, layer_name in enumerate(self.layer_name_list):
            o_prec = 32
            if ('attn_qk' in layer_name):
                w_prec = self.kv_prec
                i_prec = self.q_prec
            elif ('attn_pv' in layer_name):
                w_prec = self.kv_prec
                i_prec = self.p_prec
            else:
                w_prec = self.w_prec
                i_prec = self.a_prec

            w_dim = self.weight_dim[layer_name]
            i_dim = self.input_dim[layer_name]
            o_dim = self.output_dim[layer_name]

            # batch_size, output channel, weight hidden size
            batch_kv, cout_w, cin_w = w_dim
            # batch size, num token, input hidden size
            batch_size_in, num_token_in, cin_i = i_dim
            # batch size, num token, output hidden size
            batch_size_out, num_token_out, cin_o = o_dim
            assert cin_w == cin_i, f'The last dimension of weight and input matrices, {cin_w} and {cin_i}, do not match.'
            assert cout_w == cin_o, f'The output dimension of weight and output matrices, {cout_w} and {cin_o}, do not match.'
            assert num_token_in == num_token_out, f'The num_token of input and output matrices, {num_token_in} and {num_token_out}, do not match.'
            assert batch_size_in == batch_size_out, f'The batch_size of input and output matrices, {batch_size_in} and {batch_size_out}, do not match.'
            
            self._w_mem_byte_required[layer_name] = math.ceil(cin_w * w_prec / 8) * cout_w * batch_kv
            self._i_mem_byte_required[layer_name] = math.ceil(cin_i * i_prec / 8) * num_token_in * batch_size_in
            self._o_mem_byte_required[layer_name] = math.ceil(cin_o * o_prec / 8) * num_token_out * batch_size_out

    def _init_mem(self):
        ####################  DRAM  ####################
        hbm_energy_per_rw = 650
        hbm_channel_bw = 128
        num_hbm_channel = self.num_channel
        num_bank_group_per_hbm_channel = 4
        num_bank_per_hbm_bank_group = 4
        hbm_config = {
            'technology': 0.020,
            'mem_type': 'dram', 
            'size': 1e9 * 8, 
            'num_channel': num_hbm_channel, 
            'num_bank_group_per_channel': num_bank_group_per_hbm_channel, 
            'num_bank_per_group': num_bank_per_hbm_bank_group, 
            'rw_bw': hbm_channel_bw * num_hbm_channel,
            'r_port': 0, 
            'w_port': 0, 
            'rw_port': 1,
        }
        wr_cost = hbm_channel_bw / 64 * hbm_energy_per_rw
        self.dram = MemoryInstance(
            hbm_config, r_cost=wr_cost, w_cost=wr_cost, latency=1, 
            min_r_granularity=hbm_channel_bw, min_w_granularity=hbm_channel_bw, 
            get_cost_from_cacti=False, is_dram=True
        )