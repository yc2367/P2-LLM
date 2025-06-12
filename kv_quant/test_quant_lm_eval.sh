#!/bin/bash

model=("llama-3.2-3b-ins")
task_list="gsm8k_cot_llama"

# GSM8K-CoT-Llama
limit=1319 # number of samples in GSM8K dataset
num_fewshot=8
batch_size=8

#limit=256

HOME_DIR="/home/yc2367/llm/P2-LLM/kv_quant"

k_bits_list=(4)
v_bits_list=(3)
k_group_size_list=(64)
v_group_size_list=(128)
kv_residual_len_list=(1)
p_bits_pf=16
p_bits_dc=16

for kv_residual_len in "${kv_residual_len_list[@]}"  
do
    for k_bits in "${k_bits_list[@]}"
    do
        for v_bits in "${v_bits_list[@]}"
        do
            for k_group_size in "${k_group_size_list[@]}"
            do
                for v_group_size in "${v_group_size_list[@]}"  
                do
                    python ${HOME_DIR}/run_lm_eval.py --model_name ${model} \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                        --kv_quant_method "KTVT" --kv_residual_len ${kv_residual_len} --kv_quant_post_attn \
                        --tasks ${task_list} \
                        --batch_size ${batch_size} \
                        --output_dir ${HOME_DIR}/results/gsm8k_cot_llama/ \
                        --num_fewshot ${num_fewshot} --fewshot_as_multiturn --apply_chat_template \
                        --limit ${limit} \
                        --p_bits_pf ${p_bits_pf} --p_bits_dc ${p_bits_dc}

                    python ${HOME_DIR}/run_lm_eval.py --model_name ${model} \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size ${k_group_size} --v_group_size ${v_group_size} \
                        --kv_quant_method "KTVT" --kv_residual_len ${kv_residual_len} --kv_quant_post_attn --apply_k_scale \
                        --tasks ${task_list} \
                        --batch_size ${batch_size} \
                        --output_dir ${HOME_DIR}/results/gsm8k_cot_llama/ \
                        --num_fewshot ${num_fewshot} --fewshot_as_multiturn --apply_chat_template \
                        --limit ${limit} \
                        --p_bits_pf ${p_bits_pf} --p_bits_dc ${p_bits_dc}
                    
                    kv_residual_len=64
                    if [ ${k_group_size} -eq 64 ]; then
                        kv_residual_len=128;
                    fi
                    python ${HOME_DIR}/run_lm_eval.py --model_name ${model} \
                        --k_bits ${k_bits} --v_bits ${v_bits} --k_group_size $(( k_group_size*2 )) --v_group_size ${v_group_size} \
                        --kv_quant_method "KCVT" --kv_residual_len ${kv_residual_len} --kv_quant_post_attn \
                        --tasks ${task_list} \
                        --batch_size ${batch_size} \
                        --output_dir ${HOME_DIR}/results/gsm8k_cot_llama/ \
                        --num_fewshot ${num_fewshot} --fewshot_as_multiturn --apply_chat_template \
                        --limit ${limit} 
                done
            done
        done
    done
done