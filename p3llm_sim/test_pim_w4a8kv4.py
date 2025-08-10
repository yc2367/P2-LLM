import argparse

from hardware import NPU, PIM
from utils import MODEL_NAME_LIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_npu_core", type=int, default=4, help="Number of NPU cores")
    parser.add_argument("--batch_size", type=int, default=4, help="Input batch size")
    parser.add_argument("--cxt_len", type=int, default=4096, help="Input context length")

    args = parser.parse_args()
    num_npu_core  = args.num_npu_core
    batch_size    = args.batch_size
    cxt_len       = args.cxt_len

    assert batch_size > 0, "The input batch_size must be > 1"
    
    #################### Set NPU characteristic ####################
    npu_pe_energy     = 0.7
    npu_pe_array_dim  = [128, 128]
    npu_pe_dp_size    = 4

    #################### Set PIM characteristic ####################
    num_channel       = 16
    pcu_per_channel   = 8
    pe_per_pcu        = 16
    pe_dp_size        = 4
    pcu_reuse_factor  = 1

    #################### Simulate Perforamnce and Energy ####################
    total_energy_list  = [[0, 0] for _ in MODEL_NAME_LIST]
    total_latency_list = [0 for _ in MODEL_NAME_LIST]

    for idx, model_name in enumerate(MODEL_NAME_LIST):
        pim = PIM(
            model_name=model_name,
            batch_size=batch_size, 
            cxt_len=cxt_len, 
            w_prec=4, 
            a_prec=8,
            q_prec=16,
            p_prec=16,
            kv_prec=4, 
            num_channel=num_channel, 
            pcu_per_channel=pcu_per_channel, 
            pe_per_pcu=pe_per_pcu, 
            pe_dp_size=pe_dp_size, 
            pcu_reuse_factor=pcu_reuse_factor,
        )

        npu = NPU(
            model_name=model_name,
            batch_size=batch_size, 
            cxt_len=cxt_len, 
            is_prefill=False,
            w_prec=4, 
            a_prec=8,
            q_prec=16,
            p_prec=16,
            kv_prec=4, 
            num_npu_core=num_npu_core, 
            pe_dp_size=npu_pe_dp_size, 
            pe_energy=npu_pe_energy, 
            pe_array_dim=npu_pe_array_dim, 
        )
        
        ####################  Simulate  ########################################
        pim_cycle = pim.calc_cycle()
        pim_compute_energy = pim.calc_compute_energy()
        pim_output_rd_energy = pim.calc_output_rd_energy()
        npu_cycle = npu.calc_cycle()
        npu_compute_energy = npu.calc_compute_energy()
        npu_sram_rd_energy = npu.calc_sram_rd_energy()
        npu_sram_wr_energy = npu.calc_sram_wr_energy()
        npu_dram_energy    = npu.calc_dram_energy()
        #########################################################################

        total_cycle = 0
        total_energy = 0

        linear_energy = 0
        attn_energy = 0

        layer_name_list = npu.layer_name_list
        for layer_name in layer_name_list:
            if ('attn_qk' in layer_name) or ('attn_pv' in layer_name):
                layer_cycle  = npu._layer_cycle_total[layer_name]
                layer_energy = npu._layer_energy_compute[layer_name] + \
                    npu._layer_energy_sram_rd[layer_name] + \
                    npu._layer_energy_sram_wr[layer_name] + \
                    npu._layer_energy_dram[layer_name]
            else:
                layer_cycle  = pim._layer_cycle_compute[layer_name]
                layer_energy = pim._layer_energy_compute[layer_name] + pim._layer_output_rd_energy[layer_name]
            
            if ('attn_qk' in layer_name) or ('attn_pv' in layer_name):
                attn_energy += (layer_energy / 1e9)
            else:
                linear_energy += (layer_energy / 1e9)

            total_cycle  += layer_cycle
            total_energy += (layer_energy / 1e9)

        print(f'model:              {model_name}')
        print(f'total cycle:        {total_cycle}')
        total_latency_list[idx] = round(total_cycle)
        print(f'total energy:       {total_energy} mJ')
        
        print('\n')

        total_energy_list[idx][0] = round(attn_energy) 
        total_energy_list[idx][1] = round(linear_energy) 

    print(f'Latency: {total_latency_list}')
    print(f'Energy: {total_energy_list}')