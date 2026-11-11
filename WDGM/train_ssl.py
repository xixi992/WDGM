# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pywt

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from datasets.rand_conv import RandConv
from utils import utils
import vision_transformer as vits
from vision_transformer import DINOHead, MultiDINOHead
from datasets import Kinetics
from models import get_vit_base_patch16_224, get_aux_token_vit
from utils.parser import load_config
from einops import rearrange

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


# ========================= Wavelet DWT / IDWT =========================
class WaveletDWT(object):
    def __init__(self, wavelet="db1", device="cuda"):
        self.wavelet = wavelet
        self.device = device

    def dwt2(self, img: torch.Tensor):
        """
        img: [B, 3, H, W] single frame
        return LL, LH, HL, HH: each [B, 3, H//2, W//2]
        """
        B, C, H, W = img.shape
        ll_list, lh_list, hl_list, hh_list = [], [], [], []
        for b in range(B):
            for c in range(C):
                coeffs = pywt.dwt2(img[b, c].cpu().numpy(), self.wavelet)
                LL, (LH, HL, HH) = coeffs
                ll_list.append(torch.from_numpy(LL).to(self.device))
                lh_list.append(torch.from_numpy(LH).to(self.device))
                hl_list.append(torch.from_numpy(HL).to(self.device))
                hh_list.append(torch.from_numpy(HH).to(self.device))
        LL = torch.stack(ll_list).reshape(B, C, H // 2, W // 2)
        LH = torch.stack(lh_list).reshape(B, C, H // 2, W // 2)
        HL = torch.stack(hl_list).reshape(B, C, H // 2, W // 2)
        HH = torch.stack(hh_list).reshape(B, C, H // 2, W // 2)
        return LL, LH, HL, HH

    def idwt2(self, LL, LH, HL, HH):
        B, C = LL.shape[:2]
        rec_imgs = []
        for b in range(B):
            channels = []
            for c in range(C):
                coeff = (
                    LL[b, c].cpu().numpy(),
                    (LH[b, c].cpu().numpy(), HL[b, c].cpu().numpy(), HH[b, c].cpu().numpy())
                )
                rec = pywt.idwt2(coeff, self.wavelet)
                channels.append(torch.from_numpy(rec).to(self.device))
            rec_imgs.append(torch.stack(channels))
        return torch.stack(rec_imgs)


# ========================= FAGDM: Frequency‑Domain Attention‑Guided Dynamic Masking =========================
def FAGDM(attention, b, T, epoch, total_epochs, alpha_exp=0.6, p_high=0.90, p_low=0.40):
    """
    Paper formula (21)(22)(23) Bernoulli probability sampling
    :param attention: cls_attention [B, T, 196]
    :param b: batch_size_per_gpu
    :param T: frames per sample
    :param epoch: current epoch
    :param total_epochs: total training epoch
    :param alpha_exp: initial threshold coefficient 0.6
    :param p_high: mask prob for patch > gamma
    :param p_low: mask prob for patch <= gamma
    :return mask [B*T,14,14], bool True=masked
    """
    attention = attention.mean(1)  # [B,196] average over temporal dimension -> A_agg
    gamma = max(0.0, alpha_exp * (1.0 - float(epoch) / total_epochs))
    B, N_patch = attention.shape
    above_gamma = (attention > gamma)

    mask_prob = torch.where(
        above_gamma,
        torch.full_like(attention, p_high),
        torch.full_like(attention, p_low)
    )
    mask_sample = torch.bernoulli(mask_prob).detach()
    mask_attn = mask_sample.unsqueeze(1).expand(-1, T, -1).reshape(b * T, -1)
    mask = mask_attn.reshape(-1, 14, 14)
    return mask.bool()


# ========================= RTM Random Tube Masking =========================
def RTM_mask(batch_bt, num_patch_spatial=14, rho_l=0.90):
    """
    Sample spatial mask and replicate over all time steps.
    :param batch_bt: BT dimension
    :param num_patch_spatial:14 for 224/16
    :param rho_l: local view mask probability
    :return tube_mask [BT,14,14], True=masked
    """
    space_mask = torch.bernoulli(torch.full((batch_bt, num_patch_spatial, num_patch_spatial), rho_l))
    return space_mask.bool()


# ========================= Subband Reconstruction Loss: Global‑Context‑Conditioned =========================
class SubbandReconLoss(nn.Module):
    def __init__(self, embed_dim=768, lambda_LL=1.0, lambda_LH=1.0, lambda_HL=1.0, lambda_HH=1.0):
        super().__init__()
        self.proj_global_context = nn.Linear(embed_dim, embed_dim)
        self.lambdas = {"LL": lambda_LL, "LH": lambda_LH, "HL": lambda_HL, "HH": lambda_HH}
        self.out_ch = 3
        self.out_hw = 112 // 2
        total_elem = 4 * self.out_ch * self.out_hw * self.out_hw
        self.wave_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, total_elem)
        )

    def forward(self, z_global, z_local, gt_subbands_dict):
        """
        z_global:[B,N,D] global view encoder feature
        z_local:[B,N,D] masked local view encoder feature
        gt_subbands_dict:{"LL":gt_LL,"LH":gt_LH,"HL":gt_HL,"HH":gt_HH}
        return scalar reconstruction loss
        """
        # formula (26): global context pooling + layernorm
        C_global = torch.mean(z_global, dim=1)
        C_global = F.layer_norm(C_global, normalized_shape=[C_global.size(-1)])
        # formula (27): broadcast add projected global context
        C_proj = self.proj_global_context(C_global).unsqueeze(1)
        z_local_cond = z_local + C_proj

        # formula (28): predict four wavelet subbands
        pred_all = self.wave_decoder(z_local_cond)
        B = pred_all.shape[0]
        pred_all = pred_all.reshape(B, 4, self.out_ch, self.out_hw, self.out_hw)
        pred = {
            "LL": pred_all[:, 0],
            "LH": pred_all[:, 1],
            "HL": pred_all[:, 2],
            "HH": pred_all[:, 3]
        }
        loss = 0.0
        for sb in ["LL", "LH", "HL", "HH"]:
            loss += self.lambdas[sb] * F.mse_loss(pred[sb], gt_subbands_dict[sb])
        return loss


def get_args_parser():
    parser = argparse.ArgumentParser('WDGM', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
                                                                          'deit_tiny', 'deit_small', 'timesformer',
                                                                          'swin'] + torchvision_archs,
                        help="""Name of architecture to train. For quick experiments with ViTs, we recommend using 
                        vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels of input square patches - 
    default 16 (for 16x16 patches). Using smaller values leads to better performance but requires more memory. 
    Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling mixed precision training 
     (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of the DINO head output. For 
    complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head. Not normalizing 
                        leads to better performance but can make the training unstable. In our experiments, 
                        we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per‑GPU batch‑size : number of distinct images loaded on one GPU.')
    parser.add_argument('--gradient_accumulation_steps', default=3, type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--epochs', default=101, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning‑rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    # Multi‑crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.), help="""Scale range of the 
    cropped image before resizing, relatively to the origin image. Used for large global view cropping. When 
    disabling multi‑crop (--local_crops_number 0), we recommand using a wider range of scale ("--global_crops_scale 
    0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small local views to generate. 
    Set this parameter to 0 to disable multi‑crop training. When disabling multi‑crop we recommend to use 
    "--global_crops_scale 0.14 1." """)
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of small local views to 
    generate. Set this parameter to 0 to disable multi‑crop training. When disabling multi‑crop we recommend to use 
    "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="""Scale range of the 
    cropped image before resizing, relatively to the origin image. Used for small local view cropping of 
    multi‑crop.""")
    # wavelet viz
    parser.add_argument('--enable_wavelet_viz', action='store_true',
                        help='Enable wavelet visualization')
    parser.add_argument('--wt_max_samples', default=10, type=int,
                        help='Maximum samples for wavelet visualization')
    parser.add_argument('--save_dir_pth', default='./output/wavelet_visualizations', type=str,
                        help='Directory to save wavelet visualizations')
    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str, help='Please specify path to the '
                                                                                          'ImageNet training data.')
    parser.add_argument('--pretrained_rgb', default=None, type=str, help='Path to pretrained RGB model.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--save_mask_num', default="10", type=str, help='How often to save mask visualization')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; 
    see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)
    # WDGM paper hyper‑params
    parser.add_argument('--masking_ratio', default=0.90, type=float, help='[deprecated]')
    parser.add_argument('--threshold', default=0.60, type=float, help='[deprecated]')
    parser.add_argument('--alpha_exp', default=0.6, type=float, help="FAGDM initial threshold coefficient α_exp")
    parser.add_argument('--p_high', default=0.90, type=float, help="FAGDM mask prob for patch > gamma")
    parser.add_argument('--p_low', default=0.40, type=float, help="FAGDM mask prob for patch <= gamma")
    parser.add_argument('--rho_l', default=0.90, type=float, help="RTM local view mask probability")
    parser.add_argument('--alpha_g', default=0.8, type=float, help="global view LL scaling α_g")
    parser.add_argument('--alpha_l', default=1.5, type=float, help="local view high‑freq enhance α_l")
    parser.add_argument('--loss_rec_weight', default=0.2, type=float, help="subband reconstruction loss weight λ_rec")
    return parser


def train_svt(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    alpha_exp = args.alpha_exp
    p_high = args.p_high
    p_low = args.p_low
    rho_l = args.rho_l
    alpha_g = args.alpha_g
    alpha_l = args.alpha_l
    loss_rec_weight = args.loss_rec_weight
    global_crops_number = args.global_crops_number

    config = load_config(args)
    if utils.is_main_process():
        json.dump(vars(args), open(Path(args.output_dir) / "config.txt", "w"), indent=4)
    config.DATA.PATH_TO_DATA_DIR = args.data_path
    dataset = Kinetics(
        cfg=config,
        mode="train",
        num_retries=10,
        mask_ratio=0.0,
        get_flow=config.DATA.USE_FLOW,
        enable_wavelet_viz=args.enable_wavelet_viz,
        max_samples=args.wt_max_samples,
        save_dir_pth=args.save_dir_pth
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train data loaded: there are {len(dataset)} samples.")
    if config.DATA.RAND_CONV:
        rand_conv = RandConv(temporal_input=True).cuda()
    else:
        rand_conv = None

    dwt_tool = WaveletDWT(wavelet="db1", device="cuda")
    recon_criterion = SubbandReconLoss(embed_dim=768).cuda()

    args.arch = args.arch.replace("deit", "vit")
    student = None
    teacher = None
    motion_student = None
    motion_teacher = None
    motion_embed_dim = None
    motion_teacher_without_ddp = None
    embed_dim = 768

    if args.arch == "timesformer":
        if config.MODEL.TWO_TOKEN:
            student = get_aux_token_vit(cfg=config, no_head=True)
            teacher = get_aux_token_vit(cfg=config, no_head=True)
        else:
            student = get_vit_base_patch16_224(cfg=config, no_head=True)
            teacher = get_vit_base_patch16_224(cfg=config, no_head=True)
        embed_dim = student.embed_dim
        if args.pretrained_rgb is not None:
            state_dict = torch.load(args.pretrained_rgb)["teacher"]
            state_dict = {x[len("backbone."):]: y for x, y in state_dict.items() if x.startswith("backbone.")}
            msg = student.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained rgb student: {msg}")
            msg = teacher.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained rgb teacher: {msg}")
        if config.MODEL.TWO_STREAM:
            motion_student = vits.vit_small()
            motion_teacher = vits.vit_small()
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            msg = motion_student.load_state_dict(state_dict, strict=False)
            print(f"Loaded motion‑student with status: {msg}")
            msg = motion_teacher.load_state_dict(state_dict, strict=False)
            print(f"Loaded motion‑teacher with status: {msg}")
            motion_embed_dim = motion_student.embed_dim
    elif args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        raise RuntimeError(f"Unknown architecture: {args.arch}")

    if config.MODEL.TWO_STREAM or config.MODEL.TWO_TOKEN:
        student = utils.MultiCropWrapper(student, MultiDINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ))
        teacher = utils.MultiCropWrapper(
            teacher,
            MultiDINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )
    else:
        student = utils.MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ), vary_fr=config.DATA.RAND_FR)
        teacher = utils.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
            vary_fr=config.DATA.RAND_FR
        )
    if config.MODEL.TWO_STREAM:
        motion_student = utils.MultiCropWrapper(
            motion_student,
            DINOHead(motion_embed_dim, args.out_dim, args.use_bn_in_head),
        )
        motion_teacher = utils.MultiCropWrapper(
            motion_teacher,
            DINOHead(motion_embed_dim, args.out_dim, args.use_bn_in_head),
        )
        motion_student, motion_teacher = motion_student.cuda(), motion_teacher.cuda()

    if len(args.pretrained_weights) > 0:
        print(f"using pretrained weights: {args.pretrained_weights}")
        pretrained_weights = torch.load(args.pretrained_weights, map_location="cpu")
        student.load_state_dict(pretrained_weights, strict=False)
        teacher.load_state_dict(pretrained_weights, strict=False)

    student, teacher = student.cuda(), teacher.cuda()

    teacher_without_ddp = teacher
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=False)
        teacher_without_ddp = teacher.module
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)

    msg = teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    print(f"initialized teacher with student msg: {msg}")
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    if config.MODEL.TWO_STREAM:
        if utils.has_batchnorms(motion_student):
            motion_student = nn.SyncBatchNorm.convert_sync_batchnorm(motion_student)
            motion_teacher = nn.SyncBatchNorm.convert_sync_batchnorm(motion_teacher)
            motion_teacher = nn.parallel.DistributedDataParallel(motion_teacher, device_ids=[args.gpu])
            motion_teacher_without_ddp = motion_teacher.module
        else:
            motion_teacher_without_ddp = motion_teacher
        motion_student = nn.parallel.DistributedDataParallel(motion_student, device_ids=[args.gpu])
        motion_teacher_without_ddp.load_state_dict(motion_student.module.state_dict())
        for p in motion_teacher.parameters():
            p.requires_grad = False
        print(f"Motion Student and Teacher are built: they are both 2D ViT networks.")
    else:
        motion_teacher_without_ddp = None

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        global_crops=2,
        two_token=config.MODEL.TWO_TOKEN
    ).cuda()
    dino_flow_loss = None
    dino_cross_loss = None
    if config.MODEL.TWO_STREAM:
        dino_flow_loss = DINOLoss(args.out_dim, 2, args.warmup_teacher_temp,
                                  args.teacher_temp, args.warmup_teacher_temp_epochs, args.epochs).cuda()
        dino_cross_loss = DINOLoss(args.out_dim, args.local_crops_number + 2, args.warmup_teacher_temp,
                                   args.teacher_temp, args.warmup_teacher_temp_epochs, args.epochs).cuda()

    params_groups = utils.get_params_groups(student)
    recon_params_groups = utils.get_params_groups(recon_criterion)
    params_groups[0]['params'] += recon_params_groups[0]['params']
    params_groups[1]['params'] += recon_params_groups[1]['params']
    if config.MODEL.TWO_STREAM:
        motion_params_groups = utils.get_params_groups(motion_student)
        params_groups[0]['params'] += motion_params_groups[0]['params']
        params_groups[1]['params'] += motion_params_groups[1]['params']

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    else:
        raise ValueError("optimizer not supported")

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
        recon_criterion=recon_criterion
    )
    start_epoch = to_restore["epoch"]
    start_time = time.time()
    print("Starting WDGM training !")
    train_loss = []

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)
        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, dino_loss,
            data_loader, alpha_exp, p_high, p_low, rho_l, alpha_g, alpha_l, loss_rec_weight,
            optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, cfg=config, dwt_tool=dwt_tool, recon_criterion=recon_criterion,
            motion_loss=dino_flow_loss, cross_loss=dino_cross_loss,
            motion_student=motion_student, motion_teacher=motion_teacher,
            motion_teacher_without_ddp=motion_teacher_without_ddp, rand_conv=rand_conv
        )
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'motion_student': motion_student.state_dict() if motion_student is not None else 0,
            'motion_teacher': motion_teacher.state_dict() if motion_teacher is not None else 0,
            'recon_criterion': recon_criterion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        train_loss.append(train_stats['loss'])
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if epoch % args.saveckp_freq == 0 or epoch == args.epochs - 1:
            os.makedirs(f'{args.output_dir}/checkpoint', exist_ok=True)
            plt.clf()
            plt.plot(range(len(train_loss)), train_loss, label='train_loss_total', marker='o')
            plt.title('train_loss_epoch')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{args.output_dir}/checkpoint/loss_plot_epoch_{epoch:04}.png', dpi=120)
            plt.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    alpha_exp, p_high, p_low, rho_l, alpha_g, alpha_l, loss_rec_weight,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, cfg=None, dwt_tool=None, recon_criterion=None,
                    motion_teacher=None, motion_student=None,
                    motion_loss=None, cross_loss=None, motion_teacher_without_ddp=None, rand_conv=None):
    mask_viz_dir = f"{args.output_dir}/mask_visualizations/epoch_{epoch:04}"
    os.makedirs(mask_viz_dir, exist_ok=True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, (images, masks_, _, _, meta) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]
        images = [im.cuda(non_blocking=True) for im in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if cfg.MODEL.TWO_STREAM:
                student_output_rgb, student_output_flow = student(images)
                teacher_output_rgb, _ = teacher(images[:2])
                loss_cl = dino_loss(student_output_rgb, teacher_output_rgb, epoch)
                loss_total = loss_cl
            elif cfg.MODEL.TWO_TOKEN:
                student_output = student(images[2:])
                teacher_output = teacher(images[:2])
                loss_cl = dino_loss(student_output, teacher_output, epoch)
                loss_total = loss_cl
            else:
                if rand_conv is not None:
                    teacher_output, teacher_attentions = teacher([images[0], rand_conv(images[1])], return_attention=True)
                else:
                    teacher_output, teacher_attentions = teacher(images[:2], return_attention=True)

                masks_attn_list = []
                for u in range(args.global_crops_number):
                    b_per_gpu = args.batch_size_per_gpu
                    Tg = teacher_attentions[u].size(0) // b_per_gpu
                    attn = teacher_attentions[u]
                    cls_attn = attn[:, :, 0, 1:].mean(1).reshape(b_per_gpu, Tg, 196).detach()
                    mask_fagdm = FAGDM(cls_attn, b_per_gpu, Tg, epoch, args.epochs,
                                       alpha_exp=alpha_exp, p_high=p_high, p_low=p_low)
                    masks_attn_list.append(mask_fagdm)

                local_views = images[2:]
                masks_local = []
                for view in local_views:
                    bt = view.shape[0] * view.shape[2]
                    m_rtm = RTM_mask(bt, num_patch_spatial=14, rho_l=rho_l)
                    masks_local.append(m_rtm)

                if it % int(args.save_mask_num) == 0:
                    save_mask_visualization(images, masks_attn_list, epoch, it, save_dir=mask_viz_dir)

                feat_global_masked = student(images[:2], masks_attn_list, return_features=True)
                feat_local_masked = student(images[2:], masks_local, return_features=True)
                student_full_out = student(images, masks=None)
                loss_cl = dino_loss(student_full_out, teacher_output, epoch)

                # Wavelet subband prepare: use first global view, sample first batch element, first time‑frame
                B, C, T, H, W = images[0].shape
                frame_g = images[0][:, :, 0, :, :]
                LL, LH, HL, HH = dwt_tool.dwt2(frame_g)
                gt_subbands = {
                    "LL": LL * alpha_g,
                    "LH": LH * alpha_l,
                    "HL": HL * alpha_l,
                    "HH": HH * alpha_l
                }
                loss_rec = recon_criterion(z_global=feat_global_masked[0], z_local=feat_local_masked[0],
                                           gt_subbands_dict=gt_subbands)
                loss_total = loss_cl + loss_rec_weight * loss_rec

        if not math.isfinite(loss_total.item()):
            print("Loss_total is {}, stopping training".format(loss_total.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss_total.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss_total).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            if cfg.MODEL.TWO_STREAM:
                for param_q, param_k in zip(motion_student.module.parameters(), motion_teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_total.item())
        metric_logger.update(loss_cl=loss_cl.item())
        if 'loss_rec' in locals():
            metric_logger.update(loss_rec=loss_rec.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_mask_visualization(images, masks_attn_list, epoch, iteration, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    batch_idx = 0
    time_idx = 0
    for view_idx, (img_view, mask_view) in enumerate(zip(images[:2], masks_attn_list)):
        img = img_view[batch_idx, :, time_idx].cpu()
        mask = mask_view[batch_idx * len(mask_view) // len(img_view) + time_idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        H, W = img.shape[1:]
        mask_resized = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='nearest'
        ).squeeze()
        img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        mask_np = mask_resized.numpy().astype(np.uint8)
        red_mask = Image.new('RGB', (W, H), color=(255, 0, 0))
        alpha = Image.fromarray((mask_np * 128).astype(np.uint8))
        overlay = Image.composite(red_mask, img_pil, alpha)
        save_path = os.path.join(save_dir, f"epoch{epoch:03d}_iter{iteration:04d}_view{view_idx}")
        img_pil.save(f"{save_path}_original.png")
        Image.fromarray((mask_np * 255).astype(np.uint8)).save(f"{save_path}_mask.png")
        overlay.save(f"{save_path}_overlay.png")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_pil)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title("Mask (white=masked)")
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title("Image with Mask Overlay")
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_path}_grid.png", dpi=150, bbox_inches='tight')
        plt.close()


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, two_token=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.two_token = two_token
        if self.two_token:
            self.n_crops = 4
            self.global_crops = 2
            self.register_buffer("center", torch.zeros(2, out_dim))
        else:
            self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        total_loss = 0
        n_loss_terms = 0
        if self.two_token:
            student_out = [x / self.student_temp for x in student_output]
            student_out = [x.chunk(self.n_crops) for x in student_out]
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = [F.softmax((x - self.center[idx]) / temp, dim=-1) for idx, x in enumerate(teacher_output)]
            teacher_out = [x.detach().chunk(self.global_crops) for x in teacher_out]
            for iv in range(len(student_out[0])):
                if iv < 2:
                    q = teacher_out[0][0]
                    v = student_out[0][iv]
                    loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                else:
                    q = teacher_out[1][1]
                    v = student_out[1][iv]
                    loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        else:
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.n_crops)
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(self.global_crops)
            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        continue
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        if isinstance(teacher_output, (tuple, list)):
            batch_center = [torch.sum(x, dim=0, keepdim=True) for x in teacher_output]
            dist.all_reduce(batch_center[0])
            dist.all_reduce(batch_center[1])
            batch_center = [x / (len(teacher_output[0]) * dist.get_world_size()) for x in batch_center]
            self.center[0, :] = self.center[0, :] * self.center_momentum + batch_center[0] * (1 - self.center_momentum)
            self.center[1, :] = self.center[1, :] * self.center_momentum + batch_center[1] * (1 - self.center_momentum)
        else:
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WDGM', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir + "/checkpoint").mkdir(parents=True, exist_ok=True)
    train_svt(args)
