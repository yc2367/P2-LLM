#!/bin/bash

SCRIPTS_DIR="/home/yc2367/llm/P2-LLM/wkva_quant/scripts"
OUTPUT_DIR=${SCRIPTS_DIR}/output
mkdir -p ${OUTPUT_DIR}

this_file=$(basename $0)
this_file=${this_file%.*}
output_file=${OUTPUT_DIR}/${this_file}.txt

if [ -f $(output_file) ]; then
    rm ${output_file}
fi
touch ${output_file}
echo ${output_file}

CUDA_VISIBLE_DEVICES=0 bash ${SCRIPTS_DIR}/test_quant_ppl_template.sh  &>  ${output_file}
