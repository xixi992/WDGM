# WDGM: Wavelet-Based Dynamic Guided Masking Framework for Endoscopic Video Analysis
This repository provides the official PyTorch implementation of the paper WDGM: Wavelet-Based Dynamic Guided Masking
Framework for Endoscopic Video Analysis


## Installation
We can install packages using provided `environment.yaml`.

```shell
cd WDGM
conda env create -f environment.yaml
conda activate WDGM
```

## Data Preparation
We use the datasets provided by [Endo-FM](https://github.com/med-air/Endo-FM) and are grateful for their valuable work.

## weights
weights:
https://pan.baidu.com/s/12D-teOiA1ttAx1cYPEFOSQ?pwd=qah4

## Pre-training
```shell
cd WDGM
wget -P checkpoints/ https://github.com/kahnchana/svt/releases/download/v1.0/kinetics400_vitb_ssl.pth
bash scripts/pretrain.sh
```

## Fine-tuning
```shell
# PolypDiag (Classification)
cd WDGM
bash scripts/eval_finetune_polypdiag.sh

# CVC (Segmentation)
cd WDGM/TransUNet
python train.py

# KUMC (Detection)
cd WDGM/STFT
bash script/train_stft.sh
```

## Acknowledgement
Our code is based on [MMCRL]( https://github.com/MLMIP/MMCRL), [Endo-FM](https://github.com/med-air/Endo-FM), [DINO](https://github.com/facebookresearch/dino), [TimeSformer](https://github.com/facebookresearch/TimeSformer), [SVT](https://github.com/kahnchana/svt), [TransUNet](https://github.com/Beckschen/TransUNet), and [STFT](https://github.com/lingyunwu14/STFT). Thanks them for releasing their codes.


## Citation
```
@article{hu2024one,
  title={ WDGM: Wavelet-Based Dynamic Guided Masking Framework for Endoscopic Video Analysis},
  author={Gelan Yang, Yaoxiang He, Yilong You, Xinni Hu},
  journal={Pattern Analysis and Applications},
  year={2026}
}
```
