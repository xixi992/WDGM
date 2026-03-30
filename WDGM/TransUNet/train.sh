#!/bin/bash

export LD_LIBRARY_PATH="/public/home/yangst/anaconda3/envs/TransUNet/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=1

python train.py