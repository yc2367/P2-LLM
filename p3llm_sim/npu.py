import math
from typing import List
from mem.mem_instance import MemoryInstance

from pe_array import SystolicArray


class NPU(SystolicArray):
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
        is_generation: bool=True,
        init_mem: bool=True,
    ): 
        super().__init__(
            model_name=model_name,
            batch_size=batch_size, 
            w_prec=w_prec, 
            a_prec=a_prec,
            q_prec=q_prec,
            p_prec=p_prec,
            kv_prec=kv_prec, 
            num_npu_core=num_npu_core, 
            pe_dp_size=pe_dp_size, 
            pe_energy=pe_energy, 
            pe_array_dim=pe_array_dim, 
            cxt_len=cxt_len, 
            is_generation=is_generation
        )

        self.cycle_compute = None
        if init_mem:
            self._init_mem()
            self._check_layer_mem_size()
            self._calc_num_mem_refetch()

    def calc_cycle(self):
        self._calc_compute_cycle()
        self._calc_dram_cycle() 
        total_cycle = 0
        total_cycle_compute = 0
        for name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute[name]
            cycle_layer_dram    = self._layer_cycle_dram[name]
            #print(f'layer name: {name}, compute: {cycle_layer_compute}, dram: {cycle_layer_dram}')
            total_cycle_compute += cycle_layer_compute
            total_cycle += max(cycle_layer_compute, cycle_layer_dram)
        self.cycle_compute = total_cycle_compute

        total_cycle_compute_linear = 0
        total_cycle_dram_linear = 0
        total_cycle_compute_attn = 0
        total_cycle_dram_attn = 0

        for name in self.layer_name_list:
            if ('attn_qk' in name) or ('attn_pv' in name):
                total_cycle_compute_attn += self._layer_cycle_compute[name]
                total_cycle_dram_attn += self._layer_cycle_dram[name]
            else:
                total_cycle_compute_linear += self._layer_cycle_compute[name]
                total_cycle_dram_linear += self._layer_cycle_dram[name]
        print(f'Linear Compute: {total_cycle_compute_linear}, Linear DRAM: {total_cycle_dram_linear}')
        print(f'Attn Compute:   {total_cycle_compute_attn}, Attn DRAM:   {total_cycle_dram_attn}')
        print('\n')
        
        return total_cycle_compute, total_cycle
    
    def _calc_compute_cycle(self):
        self._layer_cycle_compute = {}
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]

            if w_dim is not None:
                cycle_layer_compute = self._calc_tile_fc(w_dim, o_dim)
                self._layer_cycle_compute[name] = cycle_layer_compute

        return

    def _calc_tile_fc(self, w_dim, o_dim):
        pe_dp_size = self.pe_dp_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # output channel, input channel
        _, cout, cin = w_dim
        # num token, output channel
        batch_size, num_token, _ = o_dim

        # tile_cin:   number of tiles along input channel
        # tile_cout:  number of tiles along output channel
        tile_cin   = math.ceil(cin / pe_dp_size)
        tile_cout  = math.ceil(cout / num_pe_row)
        tile_token = math.ceil(num_token / num_pe_col)
        total_tile = (tile_cin * tile_cout * tile_token) * batch_size

        return total_tile
    
    def _calc_dram_cycle(self):
        self._layer_cycle_dram = {}
        dram_bw = self.dram.rw_bw 

        for name in self.layer_name_list:
            num_dram_fetch_w, num_dram_fetch_i = self._layer_mem_refetch[name]
            cycle_dram_load_w = self._w_mem_byte_required[name] * 8 / dram_bw
            cycle_dram_load_w *= num_dram_fetch_w
            cycle_dram_load_i = self._i_mem_byte_required[name] * 8 / dram_bw 
            cycle_dram_load_i *= num_dram_fetch_i
            cycle_dram_write_o = self._o_mem_byte_required[name] * 8 / dram_bw
            
            cycle_layer_dram = cycle_dram_load_w + cycle_dram_write_o + cycle_dram_load_i
            self._layer_cycle_dram[name] = int(cycle_layer_dram)

        return
    
    def calc_compute_energy(self):
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        compute_energy = self.pe_energy * self.total_pe_count * self.cycle_compute
        return compute_energy
    
    def calc_sram_rd_energy(self):
        w_sram_rd_cost = self.w_sram.r_cost
        i_sram_rd_cost = self.i_sram.r_cost

        total_tile = 0
        for layer_name in self.layer_name_list:
            w_dim = self.weight_dim[layer_name]
            o_dim = self.output_dim[layer_name]
            total_tile += self._calc_tile_fc(w_dim, o_dim)

        sram_rd_energy = total_tile * (w_sram_rd_cost + i_sram_rd_cost)
        return sram_rd_energy
    
    def calc_sram_wr_energy(self):
        total_energy = 0
        for layer_name in self.layer_name_list:
            total_energy += self._calc_sram_wr_energy_fc(layer_name)
        return total_energy
    
    def _calc_sram_wr_energy_fc(self, layer_name):
        w_dim = self.weight_dim[layer_name]
        i_dim = self.input_dim[layer_name]
        o_dim = self.output_dim[layer_name]

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

        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # batch_size, output channel, weight hidden size
        batch_kv, cout_w, cin_w = w_dim
        # batch size, num token, input hidden size
        batch_size_in, num_token_in, cin_i = i_dim
        # batch size, num token, output hidden size
        batch_size_out, num_token_out, cin_o = o_dim

        # write energy, read from DRAM and write to SRAM
        num_w_sram_wr    = math.ceil(cin_w * w_prec / w_sram_min_wr_bw) * cout_w * batch_kv
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w
        num_i_sram_wr    = math.ceil(cin_i * i_prec / i_sram_min_wr_bw) * num_token_in * batch_size_in
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i
        num_o_sram_wr    = math.ceil(cin_o * o_prec / i_sram_min_wr_bw) * num_token_out * batch_size_out
        energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy
    
    def calc_dram_energy(self):
        energy = 0
        for name in self.layer_name_list:
            energy += self._calc_dram_energy_fc(name)
        return energy
    
    def _calc_dram_energy_fc(self, layer_name):
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # energy_weight: energy to read weight from DRAM
        w_mem_required = self._w_mem_byte_required[layer_name]
        energy_weight = w_mem_required * 8 / bus_width * rd_cost
        # energy_input:  energy to read input feature from DRAM
        i_mem_required = self._i_mem_byte_required[layer_name]
        energy_input  = i_mem_required * 8 / bus_width * rd_cost
        # energy_output: energy to write output feature to DRAM
        o_mem_required = self._o_mem_byte_required[layer_name]
        energy_output = o_mem_required * 8 / bus_width * wr_cost

        energy_weight *= num_fetch_w
        energy_input  *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
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

    def _calc_num_mem_refetch(self):
        # If the on-chip buffer size is not big enough, 
        # we need to refetch input tiles or weight tiles from DRAM
        self._layer_mem_refetch = {}
        size_sram_w   = self.w_sram.size / 8
        size_sram_i   = self.i_sram.size / 8
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            if w_dim is not None:
                w_mem_required = self._w_mem_byte_required[name]
                i_mem_required = self._i_mem_byte_required[name]
                if ( w_mem_required > size_sram_w ) and ( i_mem_required > size_sram_i ):
                    # need DRAM refetch
                    num_refetch_input  = math.ceil(w_mem_required / size_sram_w)
                    num_refetch_weight = math.ceil(i_mem_required / size_sram_i)
                    total_fetch_weight = num_refetch_weight * w_mem_required
                    total_fetch_input  = num_refetch_input * i_mem_required
                    if ( total_fetch_weight + i_mem_required ) < ( total_fetch_input + w_mem_required ):
                        # refetch all weight for every input tile
                        self._layer_mem_refetch[name] = (num_refetch_weight, 1)
                    else:
                        # refetch all input for every weight tile
                        self._layer_mem_refetch[name] = (1, num_refetch_input)
                else:
                    # no need refetch
                    self._layer_mem_refetch[name] = (1, 1)
        return

    def _init_mem(self):
        ####################  Weight Scratchpad  ####################
        '''
            By default, the weight scratchpad contains 32 banks, with a total capacity of 8-MiB per NPU core.
        '''
        w_bandwidth = self.pe_dp_size * self.w_prec * self.pe_array_dim['h']
        w_sram_bank = 32
        w_sram_size_per_npu = 8 * 1024**2
        num_bits_per_byte = 8
        w_sram_config = {
            'technology': 0.028,
            'mem_type': 'ram', 
            'size': self.num_npu_core * w_sram_size_per_npu * num_bits_per_byte, 
            'bank_count': w_sram_bank, 
            'rw_bw': w_bandwidth, 
            'r_port': 1, 
            'w_port': 1, 
            'rw_port': 0,
        }
        self.w_sram = MemoryInstance(
            w_sram_config, r_cost=0, w_cost=0, latency=1, 
            min_r_granularity=None, min_w_granularity=64, 
            get_cost_from_cacti=True
        )
        
        ####################  Input Scratchpad  ####################
        '''
            By default, the activation scratchpad contains 32 banks, with a total capacity of 8-MiB per NPU core.
        '''
        i_bandwidth = self.pe_dp_size * self.a_prec * self.pe_array_dim['w']
        i_sram_bank = 32
        i_sram_size_per_npu = 8 * 1024**2
        num_bits_per_byte = 8
        i_sram_config = {
            'technology': 0.028,
            'mem_type': 'ram', 
            'size': self.num_npu_core * i_sram_size_per_npu * num_bits_per_byte, 
            'bank_count': i_sram_bank, 
            'rw_bw': i_bandwidth,
            'r_port': 1, 
            'w_port': 1, 
            'rw_port': 0,
        }
        self.i_sram = MemoryInstance(
            i_sram_config, r_cost=0, w_cost=0, latency=1, 
            min_r_granularity=64, min_w_granularity=64, 
            get_cost_from_cacti=True
        )
        
        hbm_energy_per_rw = 650
        hbm_channel_bw = 128
        num_hbm_channel = 16
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
            get_cost_from_cacti=False
        )