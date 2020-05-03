#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import sys
import json
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path

import torch
from torchvision import models

sys.path.append(str(Path(sys.path[0]).parent))
from common.transforms import get_test_transform, ToTensor
from common.utils import postprocessing, draw_bboxes
from common.evaluator import Evaluator


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(
        description='Demo application for vertebra classification.'
    )
    parser.add_argument('--images', type=Path, required=True,
                        help='Path to dir with images for classification. '
                             'It also can be a markup file for evaluation.')
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path to trained model or dir with models.')
    parser.add_argument('--save-to', type=Path,
                        help='Path to dir for store classification results '
                             'instead of displaying them.')
    parser.add_argument('--iou-th', type=float, default=0.5,
                        help='Arg for remove intersecting boxes.')

    return parser.parse_args()


def read_model(model_path, device):
    """Read and prepare model and transforms for prediction."""
    if model_path.suffix == '.pth':
        ckpt = torch.load(model_path, map_location=device)
        model = models.detection.fasterrcnn_resnet50_fpn(num_classes=3)
        model.load_state_dict(ckpt['model'])

        rss = ckpt['args'].resize_size
        ccs = ckpt['args'].center_crop_size
        mv = ckpt['args'].min_visibility
        mean = ckpt['args'].mean
        std = ckpt['args'].std

    elif model_path.suffix == '.zip':
        model = torch.jit.load(str(model_path))

        rss = model.resize_size
        ccs = model.center_crop_size
        mv = model.min_visibility
        mean = model.mean
        std = model.std

    else:
        raise AttributeError('wrong model type (expected .pth or .zip ,'
                             ' got {}.'.format(model_path.suffix))

    model.to(device)
    model.eval()

    vis_transform, _ = get_test_transform(rss, ccs, mv)
    to_tensor_transform = ToTensor(normalize={'mean': mean, 'std': std})

    return model, vis_transform, to_tensor_transform


def main():
    """Application entry point."""
    args = get_args()
    if args.save_to is not None:
        args.save_to.mkdir(exist_ok=True, parents=True)

    if args.images.is_file():
        markup = True
        evaluator = Evaluator()

        with open(str(args.images)) as f:
            samples = json.load(f)

    else:
        markup = False
        evaluator = None
        samples = [str(x) for x in args.images.glob('*')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare models paths
    if args.model_path.is_file():
        models_paths = [args.model_path]
    else:
        models_paths = list(args.model_path.glob('*'))

    for model_path in models_paths:
        model, vis_transform, to_tensor_transform = read_model(
            model_path, device
        )

        cpu_device = torch.device('cpu')
        with torch.no_grad():
            for sample in tqdm(samples, desc='Predicting'):
                # get and process output
                img_path = sample['img_path'] if markup else sample
                img = cv2.imread(img_path)
                if markup:
                    sample = {
                        'image': img,
                        'bboxes': [x['bbox'] for x in sample['annotation']],
                        'labels': [int(x['label'])
                                   for x in sample['annotation']]
                    }
                else:
                    sample = {'image': img, 'bboxes': [], 'labels': []}

                # prepare image for visualization
                sample = vis_transform(**sample)
                img = sample['image'].copy()

                # prepare sample for model applying
                sample = to_tensor_transform(**sample)
                sample['image'] = sample['image'].to(device)

                # applying model
                output = model([sample['image']])

                # for fasterrcnn from torchscript it returns [losses, outputs]
                if len(output) == 2:
                    output = output[1]

                output = {k: v.to(cpu_device).numpy()
                          for k, v in output[0].items()}
                output = postprocessing([output], iou_th=args.iou_th)[0]

                drawn_img = draw_bboxes(
                    img.copy(), output['boxes'], output['labels'],
                    shifted_labels=True
                )

                # evaluate if markup file was passed
                if markup:
                    sample['boxes'] = sample['bboxes']
                    evaluator.collect_stats([output], [sample])

                    gt_drawn_img = draw_bboxes(
                        img,
                        sample['bboxes'],
                        sample['labels'],
                        shifted_labels=True
                    )
                    drawn_img = np.concatenate([gt_drawn_img, drawn_img],
                                               axis=1)

                if args.save_to is None:
                    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                    cv2.imshow('img', drawn_img)
                    cv2.waitKey()

                else:
                    model_save_folder = args.save_to.joinpath(
                        model_path.stem + '_preds')
                    model_save_folder.mkdir(exist_ok=True, parents=True)

                    save_path = model_save_folder.joinpath(Path(img_path).name)
                    cv2.imwrite(str(save_path), drawn_img)

        # calculate metrics if markup file was passed
        if markup:
            print('\nMetrics for model:', model_path.name)
            metrics = evaluator.calculate_metrics()
            for name, value in metrics.items():
                print('{}: {}'.format(name, value))


if __name__ == '__main__':
    main()
