#!/bin/bash

########## Modify the path according to your HOME directory ##########
HOME_DIR="/home/yc2367/llm/P2-LLM/wkva_quant"
AWQ_DIR="/share/abdelfattah/temp_yc2367/awq_quant_model"
######################################################################

OUTPUT_DIR=${HOME_DIR}/results/ppl

model_list=("llama-3.1-8b" "llama-3.2-3b" "mistral-7b")

dataset_list="wikitext,c4"

k_bits_list=(4)
v_bits_list=(4)
k_group_size_list=(128)
v_group_size_list=(128)

p_bits_list=(16)

w_bits_list=(4)
w_group_size_list=(128)

a_bits=(8)
a_group_size_list=(-1)


for model_name in "${model_list[@]}"
do
    for k_bits in "${k_bits_list[@]}"
    do
        for v_bits in "${v_bits_list[@]}"
        do
            for k_group_size in "${k_group_size_list[@]}"
            do
                for v_group_size in "${v_group_size_list[@]}"  
                do
                    for w_bits in "${w_bits_list[@]}"
                    do
                        for w_group_size in "${w_group_size_list[@]}"
                        do
                            for p_bits in "${p_bits_list[@]}"
                            do
                                for a_group_size in "${a_group_size_list[@]}"  
                                do
                                    ####################  All FP16  ####################
                                    python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                        --use_fp16 \
                                        --datasets ${dataset_list} --seq_len 2048 \
                                        --output_dir ${OUTPUT_DIR}
                                    
                                    ####################  Activation Quant  ####################
                                    python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                        --datasets ${dataset_list} --seq_len 2048 \
                                        --output_dir ${OUTPUT_DIR} \
                                        --a_bits 8 
                                    
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done