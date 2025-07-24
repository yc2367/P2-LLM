#!/bin/bash

########## Modify the path according to your HOME directory ##########
HOME_DIR="/home/yc2367/llm/P2-LLM/wkva_quant"
AWQ_DIR="/share/abdelfattah/temp_yc2367/awq_quant_model"
######################################################################

OUTPUT_DIR=${HOME_DIR}/results/ppl_motivation

model_list=("llama-3.2-3b")

dataset_list="wikitext,c4"

k_bits_list=(8 7 6 5 4 3)
v_bits_list=(4)
k_group_size_list=(128)
v_group_size_list=(128)

p_bits_list=(8 7 6 5 4)

w_bits_list=(8 7 6 5 4 3)
w_bits_list=(8)
w_group_size_list=(128)

a_bits_list=(8 7 6 5 4)
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
                                for a_bits in "${a_bits_list[@]}"
                                do
                                    for a_group_size in "${a_group_size_list[@]}"  
                                    do
                                        if [ ${w_bits} = 8 ] 
                                        then
                                            w_group_size=256
                                        fi

                                        ####################  All FP16  ####################
                                        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                            --use_fp16 \
                                            --datasets ${dataset_list} --seq_len 2048 \
                                            --output_dir ${OUTPUT_DIR}
                                        
                                        ####################  KV-cache quant  ####################
                                        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                            --datasets ${dataset_list} --seq_len 2048 \
                                            --output_dir ${OUTPUT_DIR} \
                                            --kv_quant_method "KTVT" \
                                            --k_bits ${k_bits} --v_bits ${k_bits} --k_group_size 128 --v_group_size 128 \
                                        
                                        #################### Weight Quant ####################
                                        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                            --datasets ${dataset_list} --seq_len 2048 \
                                            --output_dir ${OUTPUT_DIR} \
                                            --w_bits ${w_bits} --w_group_size ${w_group_size} \
                                            --awq_model_path_lp ${AWQ_DIR}/${model_name}/w${w_bits}-g${w_group_size}
                                        
                                        #################### Activation quant  ####################
                                        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                            --datasets ${dataset_list} --seq_len 2048 \
                                            --output_dir ${OUTPUT_DIR} \
                                            --a_bits ${a_bits}
                                        
                                        #################### Attention Score quant  ####################
                                        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                                            --datasets ${dataset_list} --seq_len 2048 \
                                            --output_dir ${OUTPUT_DIR} \
                                            --p_bits ${p_bits}
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done