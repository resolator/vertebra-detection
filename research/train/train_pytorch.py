#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import configargparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torch.utils.data import DataLoader

from vertebra_dataset import *


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(
        description='Train Faster RCNN on pytorch.'
    )
    parser.add_argument('--config', is_config_file=True,
                        help='Path to config file.')
    parser.add_argument('--train-json', required=True,
                        help='Path to train.json file.')
    parser.add_argument('--test-json', required=True,
                        help='Path to train.json file.')
    parser.add_argument('--save-to', required=True,
                        help='Path to saving dir.')
    parser.add_argument('--bs', type=int, default=2,
                        help='Size of batch.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--crop-factor', type=float, default=0.1,
                        help='Crop factor part of image\'s top-left.')
    parser.add_argument('--h-flip-prob', type=float, default=0.5,
                        help='Probability for horizontal flip.')
    parser.add_argument('--v-flip-prob', type=float, default=0.5,
                        help='Probability for vertical flip.')
    parser.add_argument('--mean', nargs=3, type=float,
                        default=[0.15790240544637338,
                                 0.1552071038276819,
                                 0.14960934155744626],
                        help='Mean for normalization.')
    parser.add_argument('--std', nargs=3, type=float,
                        default=[0.17360058569013256,
                                 0.1712755483385491,
                                 0.16214239162025465],
                        help='STD for normalization.')
    parser.add_argument('--loader-workers', type=int, default=1,
                        help='Num of threads for DataLoaders.')
    parser.add_argument('--pretrained-backbone', action='store_true',
                        help='Load weights for net backbone.')

    args = parser.parse_args()

    os.makedirs(args.save_to, exist_ok=True)

    return args


def main():
    """Application entry point."""
    args = get_args()

    # data processing preparation
    train_transforms = transforms.Compose([
        BottomRightCrop(args.crop_factor),
        Resize(224, 224),
        RandomHorizontalFlip(args.h_flip_prob),
        RandomVerticalFlip(args.v_flip_prob),
        ToTensor(),
        Normalize(mean=args.mean, std=args.std)
    ])
    test_transforms = transforms.Compose([
        BottomRightCrop(args.crop_factor),
        Resize(224, 224),
        ToTensor(),
        Normalize(mean=args.mean, std=args.std),
    ])

    train_ds = VertebraDataset(args.train_json, transform=train_transforms)
    test_ds = VertebraDataset(args.test_json, transform=test_transforms)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.loader_workers
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.loader_workers
    )

    # model definition
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,
        num_classes=2,
        pretrained_backbone=args.pretrained_backbone
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    # criterion = nn.

    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        output = model(images, targets)
        break


if __name__ == '__main__':
    main()
