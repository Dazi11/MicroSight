# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, patches
from torchvision.transforms import transforms

import util.misc as utils
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--pre_norm', default=False, type=bool,
                        help="whether to preNorm")

    # # * Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                     help="Train segmentation head if the flag is provided")

    # Loss
    # parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
    #                     help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/public/home/gufei/wangzijun_lintao/COCOData', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='weights',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='weights/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=6, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # appendix
    parser.add_argument('--aux_loss', default=True, type=bool,
                        help="whether to preNorm")
    parser.add_argument('--eval', default=True, action='store_true')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    pathVal = "/public/home/gufei/wangzijun_lintao/COCOData/val"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 遍历文件夹中的每个文件
    for filename in os.listdir(pathVal):
        file_path = os.path.join(pathVal, filename)
        if os.path.isfile(file_path) and filename.endswith('.jpg'):
            # 打开图片文件并进行转换
            image = Image.open(file_path)
            image_tensor = transform(image)
            image_nor = normalize(image)
            print(image_nor.ndim)
            outputs = model([image_nor])
            orig_target_sizes = torch.unsqueeze(torch.stack([torch.tensor(image_tensor.shape[1]), torch.tensor(image_tensor.shape[2])], dim=0),dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            visualize_predictions(image_tensor, results[0], filename)


def visualize_predictions(image, predictions, filename):
    # fig, ax = plt.subplots(1)
    plt.figure(figsize=(16,10))
    image_array = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image_array)
    ax = plt.gca()
    # ax.imshow(image_array)
    scores = predictions["scores"].cpu()
    labels = predictions["labels"].cpu()
    bboxs = predictions["boxes"].cpu()
    n = len(bboxs)
    for i in range(n):
        bbox = bboxs[i]
        label = labels[i]
        score = scores[i]
        if score < 0.92:
            continue
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=3, edgecolor='lime',
                                 fill=False)
        ax.add_patch(rect)
        # ax.text(bbox[0], bbox[1], f'{label}: {score}', fontsize=15,
        #         bbox=dict(facecolor='yellow', alpha=0.5))
    output_path = "Prediction92/" + filename
    plt.savefig(output_path)
    # plt.imsave(output_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
