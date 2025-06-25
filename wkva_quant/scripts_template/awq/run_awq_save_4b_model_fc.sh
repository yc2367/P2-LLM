#!/bin/bash


HOME_DIR="/root/workspace/P2-LLM/3rdparty/llm-awq"
AWQ_DIR="/share/abdelfattah/temp_yc2367/awq_quant_model"

model_name_list=("llama-2-7b" "llama-2-13b" "llama-3.1-8b" "llama-3.2-3b" "llama-3.1-8b-ins" "llama-3.2-3b-ins")
w_bit_list=(4)
group_size_list=(128 64 32)


for model_name in "${model_name_list[@]}"
do
    if [[ ${model_name} == "llama-2-7b" ]]
    then
        model_path="meta-llama/Llama-2-7b-hf"
    elif [[ ${model_name} == "llama-2-13b" ]]
    then
        model_path="meta-llama/Llama-2-13b-hf"
    elif [[ ${model_name} == "llama-3.1-8b" ]]
    then
        model_path="meta-llama/Llama-3.1-8B"
    elif [[ ${model_name} == "llama-3.2-3b" ]]
    then
        model_path="meta-llama/Llama-3.2-3B"
    elif [[ ${model_name} == "llama-3.1-8b-ins" ]]
    then
        model_path="meta-llama/Llama-3.1-8B-Instruct"
    elif [[ ${model_name} == "llama-3.2-3b-ins" ]]
    then
        model_path="meta-llama/Llama-3.2-3B-Instruct"
    fi

    for w_bit in "${w_bit_list[@]}"
    do
        for group_size in "${group_size_list[@]}"
        do
            awq_cache_path=${HOME_DIR}/awq_cache/${model_name}-w${w_bit}-g${group_size}.pt
            fake_quant_save_path="${AWQ_DIR}/${model_name}/w${w_bit}-g${group_size}"

            echo 
            echo 
            echo "#################### Running Experiment ####################"
            echo "Model name        = ${model_name}"
            echo "Model path        = ${model_path}"
            echo "Quant precision   = ${w_bit}"
            echo "Quant group size  = ${group_size}"
            echo "AWQ cache path    = ${awq_cache_path}"
            echo "############################################################"
            echo 

            cd ${HOME_DIR}
            python -m awq.entry --model_path ${model_path} \
                --w_bit ${w_bit} --q_group_size ${group_size} \
                --load_awq ${awq_cache_path} --use_double_quant \
                --dump_fake ${fake_quant_save_path}
        done
    done
done
