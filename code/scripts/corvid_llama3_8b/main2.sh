#!/bin/bash

# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1

##############################
# final exp: 2024-10-12
##############################
model_max_length=2048
mm_projector_type=gate_mixer
vision_tower_mix=True
num_register_tokens=24
using_align_loss_s1=False
image_size_mix=384
# ==============================================================================
# pretraining 
# outputPT=experiments/pretrain/plain_mga_1001k-llama31_siglip_convnext-gate_mixer_hpre24_loss1
outputPT=experiments/pretrain/s2_ref_1190k-llama31_siglip_convnext-gate_mixer_hpre24_loss1
# ==============================================================================

# ==============================================================================
# Finetune
PROMPT_VERSION="llava_llama_3_1"
# sft_data=corvid_1_2m
sft_data=corvid_1m_1028

# bs=128, lr=2e-5
tbs=128
accumulation_step=8
bs=4
lr=2e-5

# *****************************************************************************
output_sft=experiments/241122-SFT_${sft_data}-s2_ref_1190k/llama31_siglip_convnext_384-gate_mixer_hpre24_loss1
mkdir -p $output_sft
mkdir -p $output_sft/result/
cp $0 $output_sft/run.sh

deepspeed llava_align/train/train_mem.py \
    --deepspeed scripts/zero2_llava_next.json \
    --model_name_or_path playground/checkpoints/Meta-Llama-3.1-8B-Instruct \
    --llm_backbone llama_3_1 \
    --llm_pad_token pad \
    --version ${PROMPT_VERSION} \
    --model_max_length $model_max_length \
    --data_path playground/data/finetune/${sft_data}.json \
    --image_folder playground/data/images \
    --vision_tower playground/checkpoints/google/siglip-so400m-patch14-384 \
    --mm_projector_type ${mm_projector_type} \
    --pretrain_mm_mlp_adapter ${outputPT}/mm_projector.bin \
    --vision_tower_mix $vision_tower_mix \
    --image_size_mix $image_size_mix \
    --num_register_tokens $num_register_tokens \
    --using_align_loss_s1 False \
    --using_align_loss_s2 False \
    --tune_align_loss_s2_estimator False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --output_dir $output_sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps $accumulation_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

# *****************************************************************************
# inference
eval_tasks='rwqa mmb_dev mmstar mmmu sqa_img ai2d mmt seed_img blink math_vista math_verse mmvp wemath m3cot pca'
echo "Evaluating LMMs: ${output_sft}"
# eval_tasks_vlmevalkit='rwqa mmb_dev mmstar mmmu sqa_img ai2d mmt seed_img blink math_vista math_verse'
max_new_token=1000
temperature=0

# **************************************************************************
cot_prompt_types='mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct mcq_direct'
# GPULIST=(0 1 2 3 4 5 6 7)
GPULIST=(0 1 2 3)
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava_align/eval/inference_llama31.py \
    --vision_tower_mix $vision_tower_mix \
    --max_new_tokens $max_new_token \
    --temperature $temperature \
    --model_path ${output_sft} \
    --eval_tasks ${eval_tasks} \
    --vqa_prompt "vqa_direct" \
    --cot_prompt_types ${cot_prompt_types} \
    --answers_file_id ${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv_mode ${PROMPT_VERSION} &
done
wait

# **************************************************************************
# eval_tasks_vlmevalkit='rwqa mmb_dev mmstar mmmu sqa_img math_vista'
cot_prompt_types='mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot mcq_cot'
# GPULIST=(0 1 2 3 4 5 6 7)
GPULIST=(0 1 2 3)
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava_align/eval/inference_llama31.py \
    --vision_tower_mix $vision_tower_mix \
    --max_new_tokens $max_new_token \
    --temperature $temperature \
    --model_path ${output_sft} \
    --eval_tasks ${eval_tasks} \
    --vqa_prompt "vqa_cot" \
    --cot_prompt_types ${cot_prompt_types} \
    --answers_file_id ${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv_mode ${PROMPT_VERSION} &
done

