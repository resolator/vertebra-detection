#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import configargparse

import numpy as np
import tensorflow as tf

from math import isfinite
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vertebra_dataset import *
from metrics import *


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
        )
        self.writer.add_summary(summary, step)


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
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs '
                             '(if not set then train forever.)')
    parser.add_argument('--bs', type=int, default=2,
                        help='Size of batch.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer.')
    parser.add_argument('--lr-decay', action='store_true',
                        help='Enable LR decay.')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer.')
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
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained FasterRCNN from torchvision.')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd',
                        help='Optimizer.')

    args = parser.parse_args()

    os.makedirs(args.save_to, exist_ok=True)

    return args


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, loader, device, ep, logger,
                    lr_scheduler=None):
    model.train()

    cycle = tqdm(loader, desc=f'Training... (Epoch #{ep})')
    ep_losses = []
    for images, targets in cycle:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        cycle.set_postfix({'Loss:': loss_value})
        ep_losses.append(loss_value)

        if not isfinite(loss_value):
            print("\n\nLoss is {}".format(loss_value))
            [print(f'{k}: {v}') for k, v in loss_dict.items()]
            # exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(ep)

    ep_loss = sum(ep_losses) / len(ep_losses)
    logger.scalar_summary('loss', ep_loss, ep)


@torch.no_grad()
def evaluate_one_epoch(model,
                       loader,
                       device,
                       ep,
                       logger,
                       m_names,
                       best_metrics,
                       iou_th=0.5):
    cpu_device = torch.device('cpu')
    model.eval()

    metrics = np.zeros(len(m_names))
    for image, target in tqdm(loader, desc=f'Testing... (Epoch #{ep})'):
        image = list(img.to(device) for img in image)

        output = model(image)

        output = [{k: v.to(cpu_device).numpy()
                   for k, v in t.items()} for t in output]

        target = [{k: v.to(cpu_device).numpy()
                   for k, v in t.items()} for t in target]

        metrics += calc_metrics(output, target, iou_th)

    # metrics preparation and dumping
    metrics_for_save = []
    for idx, (m_name, m_sum, m_best) in enumerate(zip(
            m_names, metrics, best_metrics)):
        m = m_sum / len(loader)

        print(f'Test {m_name}: {m}')
        logger.scalar_summary(m_name, m, ep)

        if m > m_best:
            best_metrics[idx] = m
            metrics_for_save.append(m_name)

    print()
    return metrics_for_save


def main():
    """Application entry point."""
    args = get_args()

    tb_dir = os.path.join(args.save_to, 'logs')
    os.makedirs(tb_dir, exist_ok=True)
    logger = Logger(tb_dir)

    # data processing preparation
    train_transforms = transforms.Compose([
        BottomRightCrop(args.crop_factor),
        RandomHorizontalFlip(args.h_flip_prob),
        RandomVerticalFlip(args.v_flip_prob),
        ToTensor(),
        Normalize(mean=args.mean, std=args.std)
    ])
    test_transforms = transforms.Compose([
        BottomRightCrop(args.crop_factor),
        ToTensor(),
        Normalize(mean=args.mean, std=args.std),
    ])

    train_ds = VertebraDataset(args.train_json, transform=train_transforms)
    test_ds = VertebraDataset(args.test_json, transform=test_transforms)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.loader_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.loader_workers,
        collate_fn=collate_fn
    )

    # model definition
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    num_classes = 2
    if args.pretrained:
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
    else:
        model = models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False, num_classes=num_classes
        )
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer definition
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(params, lr=args.lr)

    ep = 0
    if args.epochs is None:
        epochs = -1
    else:
        epochs = args.epochs

    lr_scheduler = None
    if args.lr_decay:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor
        )

    # metrics storage
    m_names = ['precision', 'recall', 'f1', 'iou', 'matched_boxes']
    best_metrics = np.zeros(len(m_names))

    # main train cycle
    while ep != epochs:
        train_one_epoch(model, optimizer, train_loader, device, ep, logger,
                        lr_scheduler)
        save_model = evaluate_one_epoch(
            model, test_loader, device, ep, logger, m_names, best_metrics)

        if len(save_model) > 0:
            for m_name in save_model:
                torch.save(
                    {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'lr_scheduler': lr_scheduler.state_dict(),
                     'args': args,
                     'epoch': ep},
                    os.path.join(args.save_to, f'{m_name}.pth')
                )
        ep += 1


if __name__ == '__main__':
    main()
