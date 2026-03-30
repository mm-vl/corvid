#!/bin/bash

sft_data=rcot_1m

torchrun --nnodes 1 --nproc_per_node 8 --master_port 29500 \
  corvid/llamav_11b/train/finetuning.py \
  --enable_fsdp \
  --lr 1e-5  \
  --num_epochs 3 \
  --batch_size_training 4 \
  --model_name playground/model/meta-llama/Llama-3.2-11B-Vision-Instruct \
  --dist_checkpoint_root_folder experiments/250214-SFT_${sft_data}/finetuned_model \
  --dist_checkpoint_folder Corvid-o1 \
  --use_fast_kernels \
  --dataset "custom_dataset" \
  --custom_dataset.test_split "test" \
  --custom_dataset.file "corvid/llamav_11b/train/datasets/cot_dataset.py" \
  --run_validation False \
  --batching_strategy padding

