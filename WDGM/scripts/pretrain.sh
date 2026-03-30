#!/bin/bash

CHECKPOINT=""

NOW_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME="${NOW_DATE}"

export LD_LIBRARY_PATH="/public/home/yangst/anaconda3/envs/MMCRL_iBOT/lib:$LD_LIBRARY_PATH"

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="$RANDOM" \
  train_wavelet_ssl.py \
  --arch "timesformer" \
  --pretrained_weights "${CHECKPOINT}" \
  --batch_size_per_gpu 2 \
  --gradient_accumulation_steps 3 \
  --data_path "./data/pretrain" \
  --output_dir "./output/pretrain/${EXP_NAME}" \
  --save_mask_num 100 \
  --epochs 30 \
  --saveckp_freq 1 \
  --opts \
  MODEL.TWO_STREAM False \
  MODEL.TWO_TOKEN False \
  DATA.NO_FLOW_AUG False \
  DATA.USE_FLOW False \
  DATA.RAND_CONV False \
  DATA.NO_SPATIAL False