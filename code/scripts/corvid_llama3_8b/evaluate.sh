#!/bin/bash

# **************************************************************************
# VLMEvalKit: rwqa mmb_dev mmstar mmmu sqa_img ai2d mmt seed_img math_vista
#eval_datasets=("rwqa" "mmb_dev" "mmstar" "mmmu" "sqa_img" "ai2d" "mmt" "seed_img" "blink" "math_vista" "math_verse" "mmvp" "wemath" "m3cot" "pca")
 eval_datasets=("ai2d")
PROMPT_VERSION="llava_llama_3_1"

# sft_data=corvid_1m
# output=experiments/241012-SFT_${sft_data}-plain_mga_1m/vilamr-llama31_siglip_convnext_384-gate_mixer

# sft_data=corvid_1m_1016
# output=experiments/241017-SFT_${sft_data}-plain_mga_1m/vilamr-llama31_siglip_convnext_384-gate_mixer_mm

sft_data=corvid_1m_0108
output=experiments/corvid/241122-SFT_${sft_data}/llama31_siglip_convnext_384-gate_mixer_hpre24_loss1
output=experiments/corvid/241122-SFT_corvid_1m_1028-s2_ref_1190k/llama31_siglip_convnext_384-gate_mixer_hpre24_loss1

echo "Testing LMMs: ${output}"
# ********************************************************************************************

# cot_prompt_type=mcq_cot
cot_prompt_type=mcq_direct

# ********************************************************************************************
# GPULIST=(0)
GPULIST=(0 1 2 3)
# GPULIST=(4 5 6 7)
# GPULIST=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPULIST[@]}

echo "CHUNKS: ${CHUNKS}"

# ********************************************************************************************
for task in "${eval_datasets[@]}"; do
    echo "task: ${task}"
    output_file=$output/result/${task}-cot_${cot_prompt_type}-merge.jsonl
    # Clear out the output file if it exists.
    > "$output_file"
    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        chunk_file="$output/result/${task}-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl"
        # 检查文件是否存在
        if [ ! -f "$chunk_file" ]; then
            # echo "Warning: File not found: $chunk_file"
            continue
        fi
        cat ${chunk_file} >> "$output_file"
    done
    wait

    python corvid/eval/calculate_acc.py \
        --result_file $output/result/${task}-cot_${cot_prompt_type}-merge.jsonl \
        --eval_task ${task}
done


# # *****************************************************************************
# # bs=128, lr=2e-5
# tbs=128
# accumulation_step=2
# bs=8
# epoch=1
# lr=2e-5

# GPULIST=(0 1 2 3 4 5 6 7)
# CHUNKS=${#GPULIST[@]}

# # **************************************************************************
# output=experiments/240916_sft-895k-ep${epoch}_bs${tbs}_lr${lr}/vilamr-llama3-8b-siglip_bunny

# benchmarks='MMT-Bench_VAL BLINK SEEDBench2_Plus MMStar'
# prompt_format=cot_detail
# # prompt_format=mcq_hint

# python corvid/eval/evaluate.py \
#     --model_path $output \
#     --benchmarks ${benchmarks} \
#     --num-chunks $CHUNKS \
#     --rule_matching True \
#     --prompt_format ${prompt_format}

# #  --judge_model none