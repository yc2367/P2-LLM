#!/bin/bash

model=("llama-3.2-3b-ins")

HOME_DIR="/home/yc2367/llm/P2-LLM/kv_quant"

k_bits_list=(4 3)
v_bits_list=(4 3)
k_group_size_list=(64 128)
v_group_size_list=(64 128)

k_bits_list=(4)
v_bits_list=(3)
k_group_size_list=(64)
v_group_size_list=(128)
p_bits_list=(16 8)

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
                    ####################  FP16  ####################
                    python ${HOME_DIR}/run_long_bench.py --model_name ${model} \
                        --use_fp16 \
                        --kv_quant_method "KTVT" --kv_quant_post_attn \
                        --output_dir ${HOME_DIR}/results/long_bench/ 
                done
            done
        done
    done
done