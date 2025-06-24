#!/bin/bash

########## Modify the path according to your HOME directory ##########
HOME_DIR="/home/yc2367/llm/P2-LLM/wkva_quant"
AWQ_DIR="/share/abdelfattah/temp_yc2367/awq_quant_model"
######################################################################

OUTPUT_DIR=${HOME_DIR}/results/arc_challenge

model_list=("llama-3.2-3b-ins" "llama-3.1-8b-ins")
task_list="arc_challenge_chat"
batch_size=16

k_bits_list=(4)
v_bits_list=(4)
k_group_size_list=(64)
v_group_size_list=(128)
kv_residual_len_list=(1 4)
p_bits_list=(8)

w_bits_list=(4)
w_group_size_list=(64 32)

a_bits=(8)
a_group_size_list=(128 -1)


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
                    for kv_residual_len in "${kv_residual_len_list[@]}"
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
                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --use_fp16 \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR}
                                        
                                        ####################  Weight FP16  ####################
                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR} \
                                            --kv_quant_method "KTVT" \
                                            --k_bits 4 --v_bits 4 --k_group_size 128 --v_group_size 128 \
                                            
                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR} \
                                            --kv_quant_method "KTVT" --apply_k_scale \
                                            --k_bits 4 --v_bits 4 --k_group_size 128 --v_group_size 128 \

                                        ####################  KV-cache FP16  ####################
                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR} \
                                            --w_bits 4 --w_group_size 64 \
                                            --awq_model_path_lp ${AWQ_DIR}/${model_name}/w4-g64
                                        
                                        ####################  KTVT  ####################
                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR} \
                                            --kv_quant_method "KTVT" --kv_residual_len ${kv_residual_len} \
                                            --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                                            --p_bits ${p_bits} \
                                            --w_bits ${w_bits} --w_group_size ${w_group_size} \
                                            --awq_model_path_lp ${AWQ_DIR}/${model_name}/w${w_bits}-g${w_group_size} \
                                            --a_bits ${a_bits} --a_group_size ${a_group_size} \
                                            --apply_w_disag --awq_model_path_lp ${AWQ_DIR}/${model_name}/w8-g256 
                                            
                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR} \
                                            --kv_quant_method "KTVT" --kv_residual_len ${kv_residual_len} \
                                            --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                                            --p_bits ${p_bits} \
                                            --w_bits ${w_bits} --w_group_size ${w_group_size} \
                                            --awq_model_path_lp ${AWQ_DIR}/${model_name}/w${w_bits}-g${w_group_size} \
                                            --a_bits ${a_bits} --a_group_size ${a_group_size} \
                                            # --apply_w_disag --awq_model_path_lp ${AWQ_DIR}/${model_name}/w8-g256 

                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR} \
                                            --kv_quant_method "KTVT" --kv_residual_len ${kv_residual_len} --apply_k_scale \
                                            --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                                            --p_bits ${p_bits} \
                                            --w_bits ${w_bits} --w_group_size ${w_group_size} \
                                            --awq_model_path_lp ${AWQ_DIR}/${model_name}/w${w_bits}-g${w_group_size} \
                                            --a_bits ${a_bits} --a_group_size ${a_group_size} \
                                            --apply_w_disag --awq_model_path_lp ${AWQ_DIR}/${model_name}/w8-g256 
                                        
                                        python ${HOME_DIR}/run_arc_challenge.py --model_name ${model_name} \
                                            --tasks ${task_list} --batch_size ${batch_size} \
                                            --output_dir ${OUTPUT_DIR} \
                                            --kv_quant_method "KTVT" --kv_residual_len ${kv_residual_len} --apply_k_scale \
                                            --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                                            --p_bits ${p_bits} \
                                            --w_bits ${w_bits} --w_group_size ${w_group_size} \
                                            --awq_model_path_lp ${AWQ_DIR}/${model_name}/w${w_bits}-g${w_group_size} \
                                            --a_bits ${a_bits} --a_group_size ${a_group_size} \
                                            # --apply_w_disag --awq_model_path_lp ${AWQ_DIR}/${model_name}/w8-g256 
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