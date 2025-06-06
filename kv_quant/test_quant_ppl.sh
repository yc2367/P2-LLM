#!/bin/bash

model=("llama-2-7b")
dataset_list="wikitext"

if [[ ${model} == *"llama-2-7b"* ]]
then
    model_path="meta-llama/Llama-2-7b-hf"
elif [[ ${model} == *"llama-2-13b"* ]]
then
    model_path="meta-llama/Llama-2-13b-hf"
elif [[ ${model} == "llama-3.2-1b" ]]
then
    model_path="meta-llama/Llama-3.2-1B"
elif [[ ${model} == "llama-3.2-3b" ]]
then
    model_path="meta-llama/Llama-3.2-3B-Instruct"
elif [[ ${model} == "llama-3.1-8b" ]]
then
    model_path="meta-llama/Llama-3.1-8B-Instruct"
elif [[ ${model} == "llama-3-8b" ]]
then
    model_path="meta-llama/Meta-Llama-3-8B"
fi

kv_bits=3
kv_group_size=64

python run_ppl.py --model_name_or_path ${model_path} \
    --datasets ${dataset_list} --seq_len 2048 \
    --k_bits ${kv_bits} --v_bits ${kv_bits} --k_group_size ${kv_group_size} --v_group_size ${kv_group_size} \
    --kv_quant_method "KTVT" --kv_residual_len ${kv_group_size} --kv_quant_post_attn

python run_ppl.py --model_name_or_path ${model_path} \
    --datasets ${dataset_list} --seq_len 2048 \
    --k_bits ${kv_bits} --v_bits ${kv_bits} --k_group_size $(( kv_group_size*2 )) --v_group_size $(( kv_group_size*2 )) \
    --kv_quant_method "KCVT" --kv_residual_len $(( kv_group_size*2 ))

python run_ppl.py --model_name_or_path ${model_path} \
    --datasets ${dataset_list} --seq_len 2048 \
    --k_bits ${kv_bits} --v_bits ${kv_bits} --k_group_size ${kv_group_size} --v_group_size ${kv_group_size} \
    --kv_quant_method "KTVT" --kv_residual_len ${kv_group_size} 
python run_ppl.py --model_name_or_path ${model_path} \
    --datasets ${dataset_list} --seq_len 2048 \
    --k_bits ${kv_bits} --v_bits ${kv_bits} --k_group_size ${kv_group_size} --v_group_size ${kv_group_size} \
    --kv_quant_method "KTVT" --kv_residual_len ${kv_group_size} --apply_k_bias
python run_ppl.py --model_name_or_path ${model_path} \
    --datasets ${dataset_list} --seq_len 2048 \
    --k_bits ${kv_bits} --v_bits ${kv_bits} --k_group_size ${kv_group_size} --v_group_size ${kv_group_size} \
    --kv_quant_method "KTVT" --kv_residual_len ${kv_group_size} --apply_k_scale
python run_ppl.py --model_name_or_path ${model_path} \
    --datasets ${dataset_list} --seq_len 2048 \
    --k_bits ${kv_bits} --v_bits ${kv_bits} --k_group_size ${kv_group_size} --v_group_size ${kv_group_size} \
    --kv_quant_method "KTVT" --kv_residual_len ${kv_group_size} --apply_k_bias --apply_k_scale