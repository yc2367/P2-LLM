#!/bin/bash

model=("llama-3.2-3b-ins")

HOME_DIR="/root/workspace/P2-LLM/kv_quant"

k_bits_list=(4)
v_bits_list=(3)
k_group_size_list=(128)
v_group_size_list=(128)
kv_residual_len=8
p_bits_list=(8)

for k_bits in "${k_bits_list[@]}"
do
    for v_bits in "${v_bits_list[@]}"
    do
        for k_group_size in "${k_group_size_list[@]}"
        do
            for v_group_size in "${v_group_size_list[@]}"  
            do
                for p_bits in "${p_bits_list[@]}"  
                do
                    ####################  KTVT  ####################
                    python ${HOME_DIR}/run_long_bench.py --model_name ${model} \
                        --kv_quant_method "KTVT" --kv_residual_len ${kv_residual_len} --kv_quant_post_attn \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                        --output_dir ${HOME_DIR}/results/long_bench/ \
                        --apply_k_scale \
                        --p_bits_pf ${p_bits} --p_bits_dc ${p_bits}
                done
            done
        done
    done
done