import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
# from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--root_path', type=str,
                    default='../data/Downstream_Data/CVC-12k/CVC-ClinicVideoDB/', help='root dir for data')
parser.add_argument('--pretrained_model_weights', type=str,
                    default='', help='path to pretrained model weights')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--output_dir', default="./output", type=str, help='Path to save logs and checkpoints.')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=9401, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference_using_eval(args, model, test_save_path=None):
    from trainer import eval

    test_transform = transforms.Compose([
        Resize((args.img_size, args.img_size)),  # 224
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ])

    db_test = SegCTDataset(dataroot=args.root_path, mode='test', transforms=test_transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    logging.info("{} test iterations per epoch".format(len(testloader)))

    if args.pretrained_model_weights and os.path.exists(args.pretrained_model_weights):
        model.load_state_dict(torch.load(args.pretrained_model_weights, map_location='cuda'), strict=False)
        logging.info(f"Loaded model weights from {args.pretrained_model_weights}")
    else:
        logging.info("No pretrained weights provided or file does not exist, using current model weights")

    model.eval()
    model.cuda()

    test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)

    logging.info('Test Dice: %.1f, HD95: %.1f' % (test_dice * 100., test_hd95))
    print('Test Dice: %.1f, HD95: %.1f' % (test_dice * 100., test_hd95))

    return "Testing Finished!"


def simple_dice(predictions, labels, num_classes=2):
    predictions = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
    dice_sum = 0
    for class_idx in range(num_classes):
        pred_class = (predictions == class_idx).float()
        true_class = (labels == class_idx).float()
        intersection = (pred_class * true_class).sum()
        union = pred_class.sum() + true_class.sum()
        dice = (2. * intersection) / (union + 1e-8)
        dice_sum += dice
    return dice_sum / num_classes


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Downstream_Data/CVC-12k/CVC-ClinicVideoDB/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    args.root_path = dataset_config[dataset_name]['volume_path']
    args.pretrained_model_weights = ""

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    # snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    # snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    # snapshot_path += '_' + args.vit_name
    # snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    # snapshot_path = snapshot_path + '_vitpatch' + str(
    #     args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    # snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    # if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
    #     snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
    #                                           0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    # snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    # snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    # snapshot_path = snapshot_path + '_' + str(args.img_size)
    # snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    snapshot = "/public/home/yangst/project/MMCRL-main_wt/TransUNet/output/model/TransUNet_2026-01-22_15-33-20/bestModel.pth"

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    print("=" * 80)
    print("Model Configuration Information:")
    print(f" Model Type: {args.vit_name}")
    print(f" Image Size: {args.img_size}")
    print(f" Number of Classes: {config_vit.n_classes}")
    print(f" Number of Skip Connections: {config_vit.n_skip}")
    print(f" hidden_size: {config_vit.hidden_size}")
    print(f" Patch Size: {config_vit.patches.size}")
    print("=" * 80)

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    print(f"Loading model weights from: {snapshot}")

    try:
        net.load_state_dict(torch.load(snapshot), strict=False)
        print("Model weights loaded successfully (with strict=False)")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Trying to load with map_location...")
        net.load_state_dict(torch.load(snapshot, map_location='cuda'), strict=False)

    snapshot_name = snapshot.split('/')[-1]

    log_folder = f'{args.output_dir}/test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    test_save_path = None

    inference_using_eval(args, net, test_save_path)