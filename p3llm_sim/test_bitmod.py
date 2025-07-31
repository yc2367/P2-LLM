import argparse

from hardware import NPU 
from utils import MODEL_NAME_LIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_npu_core", type=int, default=4, help="Number of NPU cores")
    parser.add_argument("--batch_size", type=int, default=8, help="Input batch size")
    parser.add_argument("--cxt_len", type=int, default=4096, help="Input context length")
    parser.add_argument(
        "--is_prefill", action="store_true", 
        help="If enabled, then simulate the prefill stage, otherwise the generation stage."
    )

    args = parser.parse_args()
    num_npu_core  = args.num_npu_core
    batch_size    = args.batch_size
    cxt_len       = args.cxt_len
    is_prefill = args.is_prefill

    assert batch_size > 0, "The input batch_size must be > 1"
    
    #################### Set PE array characteristic ####################
    pe_energy    = 0.7 
    pe_array_dim = [128, 128]
    pe_dp_size   = 4

    #################### Simulate Perforamnce and Energy ####################
    total_energy_list  = [[0, 0] for _ in MODEL_NAME_LIST]
    total_latency_list = [0 for _ in MODEL_NAME_LIST]

    for idx, model_name in enumerate(MODEL_NAME_LIST):
        npu = NPU(
            model_name=model_name,
            batch_size=batch_size, 
            cxt_len=cxt_len, 
            is_prefill=is_prefill,
            w_prec=4, 
            a_prec=16,
            q_prec=16,
            p_prec=16,
            kv_prec=16, 
            num_npu_core=num_npu_core, 
            pe_dp_size=pe_dp_size, 
            pe_energy=pe_energy, 
            pe_array_dim=pe_array_dim, 
        )

        total_cycle    = npu.calc_cycle()
        compute_energy = npu.calc_compute_energy() / 1e9
        sram_rd_energy = npu.calc_sram_rd_energy() / 1e9
        sram_wr_energy = npu.calc_sram_wr_energy() / 1e9
        dram_energy    = npu.calc_dram_energy() / 1e9

        sram_energy    = sram_rd_energy + sram_wr_energy
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

        print(f'model:              {model_name}')
        print(f'total cycle:        {total_cycle}')
        total_latency_list[idx] = round(total_cycle[1])

        print(f'weight buffer area: {npu.w_sram.area} mm2')
        print(f'input buffer area:  {npu.i_sram.area} mm2')
        # print(f'sram rd energy:     {sram_rd_energy} mJ')
        # print(f'sram wr energy:     {sram_wr_energy} mJ')
        print(f'dram energy:        {dram_energy} mJ')
        print(f'on-chip energy:     {onchip_energy} mJ')
        print(f'compute energy:     {compute_energy} mJ')
        print(f'total energy:       {total_energy} mJ')
        
        print('\n')

        linear_energy = 0
        attn_energy = 0
        layer_name_list = npu.layer_name_list
        for layer_name in layer_name_list:
            layer_energy = npu._layer_energy_compute[layer_name] + \
                npu._layer_energy_sram_rd[layer_name] + \
                npu._layer_energy_sram_wr[layer_name] + \
                npu._layer_energy_dram[layer_name]
            if ('attn_qk' in layer_name) or ('attn_pv' in layer_name):
                attn_energy += layer_energy
            else:
                linear_energy += layer_energy

        total_energy_list[idx][0] = round(attn_energy / 1e9) 
        total_energy_list[idx][1] = round(linear_energy / 1e9) 

    print(f'Latency: {total_latency_list}')
    print(f'Energy: {total_energy_list}')

