#!/bin/bash

# ==============================================================================
model_max_length=2048
mm_projector_type=gate_mixer
vision_tower_mix=True
num_register_tokens=0
image_size_mix=384
# ==============================================================================

##############################
# Stage I:
it_file=pretrain_plain_mga_1001k.json
PROMPT_VERSION="plain"
using_align_loss_s1=False

#tbs=256
accumulation_step=8
bs=8
lr=1e-3

pt_s1_output=experiments/pretrain/plain_mga_1001k-llama31_siglip_convnext-gate_mixer
mkdir -p $pt_s1_output
cp $0 $pt_s1_output/run.sh

# ==============================================================================
deepspeed llava_align/train/train_mem.py \
    --deepspeed scripts/zero2_llava_next.json \
    --model_name_or_path playground/checkpoints/Meta-Llama-3.1-8B-Instruct \
    --llm_backbone llama_3_1 \
    --llm_pad_token pad \
    --version ${PROMPT_VERSION} \
    --data_path playground/data/pretrain/${it_file} \
    --image_folder playground/data/images \
    --vision_tower playground/checkpoints/google/siglip-so400m-patch14-384 \
    --mm_projector_type ${mm_projector_type} \
    --vision_tower_mix $vision_tower_mix \
    --image_size_mix $image_size_mix \
    --num_register_tokens $num_register_tokens \
    --using_align_loss_s1 $using_align_loss_s1 \
    --using_align_loss_s2 False \
    --tune_align_loss_s2_estimator False \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir $pt_s1_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps $accumulation_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --model_max_length $model_max_length \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none


##############################
# Stage II:
it_file=pretrain_stage2_1190k.json
PROMPT_VERSION="llava_llama_3_1"
using_align_loss_s1=False

#tbs=256
accumulation_step=8
bs=8
lr=1e-4

pt_s2_output=experiments/pretrain/s2_ref_1190k-llama31_siglip_convnext-gate_mixer
mkdir -p $pt_s2_output
cp $0 $pt_s2_output/run.sh
# ==============================================================================
deepspeed llava_align/train/train_mem.py \
    --deepspeed scripts/zero2_llava_next.json \
    --model_name_or_path playground/checkpoints/Meta-Llama-3.1-8B-Instruct \
    --llm_backbone llama_3_1 \
    --llm_pad_token pad \
    --version ${PROMPT_VERSION} \
    --data_path playground/data/pretrain/${it_file} \
    --image_folder playground/data/images \
    --vision_tower playground/checkpoints/google/siglip-so400m-patch14-384 \
    --mm_projector_type ${mm_projector_type} \
    --pretrain_mm_mlp_adapter ${pt_s1_output}/mm_projector.bin \
    --vision_tower_mix $vision_tower_mix \
    --image_size_mix $image_size_mix \
    --num_register_tokens $num_register_tokens \
    --using_align_loss_s1 $using_align_loss_s1 \
    --using_align_loss_s2 False \
    --tune_align_loss_s2_estimator False \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir $pt_s2_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps $accumulation_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --model_max_length $model_max_length \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

