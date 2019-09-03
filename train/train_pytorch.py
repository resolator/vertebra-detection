#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import configargparse

import numpy as np
import albumentations as alb

from tqdm import tqdm
from math import isfinite, isnan

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn, faster_rcnn

from vertebra_dataset import VertebraDataset, ToTensor

sys.path.append(os.path.join(sys.path[0], '../'))
from common.metrics import calc_metrics
from common.utils import draw_bboxes, postprocessing


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
    parser.add_argument('--perspective-range', type=float, nargs=2,
                        default=[0.05, 0.15],
                        help='Warp perspective range.')
    parser.add_argument('--perspective-prob', type=float, default=0.75,
                        help='Probability of applying warp perspective.')
    parser.add_argument('--resize-size', type=int, nargs=2,
                        default=[384, 384],
                        help='Resize images to this size.')
    parser.add_argument('--rotate-range', type=int, nargs=2,
                        default=[-20, 20],
                        help='Rotate image on random angle in range.')
    parser.add_argument('--rotate-prob', type=float, default=0.75,
                        help='Probability of applying random rotation.')
    parser.add_argument('--center-crop-size', type=int, nargs=2,
                        default=[330, 330],
                        help='Size of center-cropped image.')
    parser.add_argument('--h-flip-prob', type=float, default=0.5,
                        help='Probability for horizontal flip.')
    parser.add_argument('--v-flip-prob', type=float, default=0.5,
                        help='Probability for vertical flip.')
    parser.add_argument('--brightness-contrast', type=float, default=0.1,
                        help='Randomly change brightness and contrast '
                             'of the input image.')
    parser.add_argument('--brightness-contrast-prob', type=float, default=0.75,
                        help='Probability for applying brightness '
                             'and contrast.')
    parser.add_argument('--mean', nargs=3, type=float,
                        default=[0.485, 0.456, 0.406],
                        help='Mean for normalization.')
    parser.add_argument('--std', nargs=3, type=float,
                        default=[0.229, 0.224, 0.225],
                        help='STD for normalization.')
    parser.add_argument('--loader-workers', type=int, default=1,
                        help='Num of threads for DataLoaders.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained FasterRCNN from torchvision.')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd',
                        help='Optimizer.')
    parser.add_argument('--min-visibility', type=float, default=0.4,
                        help='Minimum fraction of area for a bounding box '
                             'after augmentation to remain this box in list.')

    args = parser.parse_args()

    os.makedirs(args.save_to, exist_ok=True)

    return args


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Simple LR scheduler warmup."""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model,
                    optimizer,
                    loader,
                    device,
                    ep,
                    writer,
                    lr_scheduler=None):
    """Train model once on train dataset.

    Parameters
    ----------
    model : torch model
        FasterRCNN model for train.
    optimizer : torch optimizer
        Torch optimizer.
    loader : DataLoader
        Train dataloader.
    device : torch.device
        Device for train.
    ep : int
        Current epoch number.
    writer : SummaryWriter
        Tensorboard writter
    lr_scheduler : torch lr scheduler
        Scheduler for learrning rate.

    """
    model.train()

    ep_losses = []
    cycle = tqdm(loader, desc=f'Training {ep}')
    for batch in cycle:
        images = [x['image'].to(device) for x in batch]
        targets = [{'boxes': x['bboxes'].to(device),
                    'labels': x['labels'].to(device)} for x in batch]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        cycle.set_postfix({'Loss:': loss_value})
        ep_losses.append(loss_value)

        if not isfinite(loss_value):
            print("\n\nLoss is {}".format(loss_value))
            [print(f'{k}: {v}') for k, v in loss_dict.items()]
            exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(ep)

    ep_loss = sum(ep_losses) / len(ep_losses)
    writer.add_scalar('Loss/train', ep_loss, ep)


@torch.no_grad()
def evaluate_one_epoch(model,
                       loader,
                       device,
                       ep,
                       writer,
                       m_names,
                       best_metrics,
                       std,
                       mean,
                       metrics_iou_th=0.5,
                       iou_th_postprocessing=0.65):
    """Run model on test dataset and calculate metrics.

    Run model on all evaluation dataset. Remove very intersecting boxes from
    each prediction. Match GT and PD labels and calculate metrics on collected
    labels. Log resulted metrics and compare them with best metrics.

    Parameters
    ----------
    model : torch model
        FasterRCNN model for evaluation.
    loader : DataLoader
        Test dataloader.
    device : torch.device
        Device for evaluation.
    ep : int
        Current epoch number.
    writer : SummaryWriter
        Tensorboard writter.
    m_names : List
        Names of metrics for tensorboard .
    best_metrics : List
        List with best values of metrics.
    std : List
        Standart deviation for every channel (RGB order).
    mean : List
        Mean for every channel (RGB order).
    metrics_iou_th : float
        Threshold for match gt_box with pd_box.
    iou_th_postprocessing : float
        Threshold for filter predicted boxes.

    Returns
    -------
    List
        Metrics names for save if calculated metric is better than best.

    """
    cpu_device = torch.device('cpu')
    model.eval()

    batch_num = np.random.randint(0, len(loader))
    metrics = [[], [], [], []]
    cycle = tqdm(loader, desc=f'Testing {ep}', total=len(loader))
    for idx, batch in enumerate(cycle):
        images = [x['image'].to(device) for x in batch]
        targets = [{'boxes': x['bboxes'].to(cpu_device),
                    'labels': x['labels'].to(cpu_device)} for x in batch]

        outputs = model(images)
        outputs = [{k: v.to(cpu_device).numpy()
                    for k, v in t.items()} for t in outputs]
        outputs = postprocessing(outputs, iou_th_postprocessing)

        # metrics collecting
        for output, target in zip(outputs, targets):
            cur_m = calc_metrics(output, target, iou_th=metrics_iou_th)
            [m.append(x) for x, m in zip(cur_m, metrics) if not isnan(x)]

        # draw random image
        if idx == batch_num:
            sample_idx = np.random.randint(0, len(images))

            gt_img = draw_bboxes(
                images[sample_idx],
                targets[sample_idx]['boxes'],
                targets[sample_idx]['labels'],
                shifted_labels=True,
                mean=mean,
                std=std
            )
            pd_img = draw_bboxes(
                images[sample_idx],
                outputs[sample_idx]['boxes'],
                outputs[sample_idx]['labels'],
                shifted_labels=True,
                mean=mean,
                std=std
            )
            writer.add_image('Test/gt_image', gt_img, ep, dataformats='HWC')
            writer.add_image('Test/pd_image', pd_img, ep, dataformats='HWC')

    # metrics preparation and dumping
    metrics_for_save = ['last']
    for idx, (m_name, m, m_best) in enumerate(
            zip(m_names, metrics, best_metrics)):
        m = np.mean(m)
        print(f'Test mean {m_name}: {m} (best is {m_best})')
        writer.add_scalar('Test/mean_' + m_name, m, ep)

        if m > m_best:
            best_metrics[idx] = m
            metrics_for_save.append(m_name)

    print()
    return metrics_for_save


def main():
    """Application entry point."""
    args = get_args()

    logs_path = os.path.join(args.save_to, 'logs')
    os.makedirs(logs_path, exist_ok=True)
    writer = SummaryWriter(logs_path)

    # data processing preparation
    bbox_params = alb.BboxParams(
        format='pascal_voc',
        min_visibility=args.min_visibility,
        label_fields=['labels']
    )

    train_transforms = alb.Compose([
        alb.IAAPerspective(args.perspective_range, keep_size=False,
                           p=args.perspective_prob),
        alb.Resize(args.resize_size[0], args.resize_size[1]),
        alb.Rotate(args.rotate_range, p=args.rotate_prob),
        alb.RandomRotate90(),
        alb.HorizontalFlip(p=args.h_flip_prob),
        alb.VerticalFlip(p=args.v_flip_prob),
        alb.CenterCrop(args.center_crop_size[0], args.center_crop_size[1]),
        alb.RandomBrightness(args.brightness_contrast_prob,
                             p=args.brightness_contrast_prob),
        ToTensor(normalize={'mean': args.mean, 'std': args.std})
    ], bbox_params=bbox_params)

    test_transforms = alb.Compose([
        alb.Resize(args.resize_size[0], args.resize_size[1]),
        alb.CenterCrop(args.center_crop_size[0], args.center_crop_size[1]),
        ToTensor(normalize={'mean': args.mean, 'std': args.std})
    ], bbox_params=bbox_params)

    train_ds = VertebraDataset(args.train_json, transform=train_transforms)
    test_ds = VertebraDataset(args.test_json, transform=test_transforms)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.loader_workers,
        collate_fn=train_ds.collate_fn
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.loader_workers,
        collate_fn=test_ds.collate_fn
    )

    # model definition
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    num_classes = 3
    if args.pretrained:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    else:
        model = fasterrcnn_resnet50_fpn(
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

    lr_s = None
    if args.lr_decay:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_s = warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor
        )

    # metrics storage
    m_names = ['precision', 'recall', 'f1', 'AP']
    best_metrics = np.zeros(len(m_names))

    models_path = os.path.join(args.save_to, 'models')
    os.makedirs(models_path, exist_ok=True)

    # main train cycle
    while ep != epochs:
        train_one_epoch(model, optimizer, train_loader, device, ep, writer,
                        lr_s)
        save_model = evaluate_one_epoch(
            model,
            test_loader,
            device,
            ep,
            writer,
            m_names,
            best_metrics,
            args.std,
            args.mean
        )

        for m_name in save_model:
            torch.save(
                {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'lr_scheduler': lr_s if lr_s is None else lr_s.state_dict(),
                 'args': args,
                 'epoch': ep},
                os.path.join(models_path, f'{m_name}.pth')
            )
        ep += 1


if __name__ == '__main__':
    main()
