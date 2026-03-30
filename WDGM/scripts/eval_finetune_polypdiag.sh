#!/bin/bash

CHECKPOINT=""
DATA_PATH="data/Downstream_Data/PolypDiag"

NOW_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME="${NOW_DATE}"

DATASET="ucf101"

export LD_LIBRARY_PATH="/public/home/yangst/anaconda3/envs/MMCRL_wt/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

if [ ! -d "checkpoints/${EXP_NAME}" ]; then
  mkdir "output/polypdiag/eval/${EXP_NAME}"
fi

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_finetune.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "${CHECKPOINT}" \
  --epochs 20 \
  --lr 0.001 \
  --batch_size_per_gpu 4 \
  --num_workers 4 \
  --num_labels 2 \
  --dataset "${DATASET}" \
  --output_dir "./output/polypdiag/eval/${EXP_NAME}" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False