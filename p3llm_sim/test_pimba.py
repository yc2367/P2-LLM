import argparse

from hardware import PIM 
from utils import MODEL_NAME_LIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Input batch size")
    parser.add_argument("--cxt_len", type=int, default=16384, help="Input context length")

    args = parser.parse_args()
    batch_size    = args.batch_size
    cxt_len       = args.cxt_len

    assert batch_size > 0, "The input batch_size must be > 1"
    
    #################### Set PIM characteristic ####################
    num_channel       = 16
    pcu_per_channel   = 8
    pe_per_pcu        = 16
    pe_dp_size        = 1
    pcu_reuse_factor  = 1

    #################### Simulate Perforamnce and Energy ####################
    total_energy_list  = [[0, 0] for _ in MODEL_NAME_LIST]
    total_energy_list  = [0 for _ in MODEL_NAME_LIST]
    total_latency_list = [0 for _ in MODEL_NAME_LIST]

    for idx, model_name in enumerate(MODEL_NAME_LIST):
        acc = PIM(
            model_name=model_name,
            batch_size=batch_size, 
            cxt_len=cxt_len, 
            w_prec=8, 
            a_prec=8,
            q_prec=8,
            p_prec=8,
            kv_prec=8, 
            num_channel=num_channel, 
            pcu_per_channel=pcu_per_channel, 
            pe_per_pcu=pe_per_pcu, 
            pe_dp_size=pe_dp_size, 
            pcu_reuse_factor=pcu_reuse_factor,
        )

        total_cycle      = acc.calc_cycle()
        compute_energy   = acc.calc_compute_energy() / 1e9
        output_rd_energy = acc.calc_output_rd_energy() / 1e9
        total_energy   = compute_energy + output_rd_energy 

        print(f'model:              {model_name}')
        print(f'total cycle:        {total_cycle}')
        total_latency_list[idx] = round(total_cycle)

        print(f'compute energy:     {compute_energy} mJ')
        print(f'total energy:       {total_energy} mJ')
        # total_energy_list[idx][0] = round(output_rd_energy)
        # total_energy_list[idx][1] = round(total_energy)
        total_energy_list[idx] = round(total_energy)
        
        print('\n')

    print(f'Latency: {total_latency_list}')
    print(f'Energy: {total_energy_list}')
