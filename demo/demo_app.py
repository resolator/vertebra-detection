#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import argparse

import numpy as np

from tqdm import tqdm

import torch
from torchvision import models

sys.path.append(os.path.join(sys.path[0], '../'))
from common.transforms import get_test_transform, ToTensor
from common.utils import postprocessing, draw_bboxes
from common.evaluator import Evaluator


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(
        description='Demo application for vertebra classification.'
    )
    parser.add_argument('--images', required=True,
                        help='Path to dir with images for classification. '
                             'It also can be a markup file for evaluation.')
    parser.add_argument('--model-path', required=True,
                        help='Path to trained model or dir with models.')
    parser.add_argument('--save-to',
                        help='Path to dir for store classification results '
                             'instead of displaying them.')
    parser.add_argument('--iou-th', type=float, default=0.5,
                        help='Arg for remove intersecting boxes.')

    args = parser.parse_args()
    if args.save_to is not None:
        os.makedirs(args.save_to, exist_ok=True)

    return args


def main():
    """Application entry point."""
    args = get_args()

    markup = False
    evaluator = None
    if os.path.isfile(args.images):
        markup = True
        evaluator = Evaluator()

        with open(args.images) as f:
            samples = json.load(f)

    else:
        samples = [os.path.join(args.images, path)
                   for path in os.listdir(args.images)]

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if os.path.isfile(args.model_path):
        models_paths = [args.model_path]
    else:
        models_paths = [os.path.join(args.model_path, name)
                        for name in os.listdir(args.model_path)]

    for model_path in models_paths:
        # model preparation
        ckpt = torch.load(model_path, map_location=device)

        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                         num_classes=3)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()

        # create transform for visualization only (without ToTensor)
        vis_transform, _ = get_test_transform(
            ckpt['args'].resize_size,
            ckpt['args'].center_crop_size,
            ckpt['args'].min_visibility
        )
        to_tensor_transform = ToTensor(
            normalize={'mean': ckpt['args'].mean, 'std': ckpt['args'].std}
        )

        cpu_device = torch.device('cpu')
        with torch.no_grad():
            for sample in tqdm(samples, desc='Predicting'):
                # get and process output
                if markup:
                    img_path = sample['img_path']
                else:
                    img_path = sample

                img = cv2.imread(img_path)
                if markup:
                    sample = {
                        'image': img,
                        'bboxes': [x['bbox'] for x in sample['annotation']],
                        'labels': [int(x['label'])
                                   for x in sample['annotation']]
                    }
                else:
                    sample = {
                        'image': img,
                        'bboxes': [],
                        'labels': []
                    }

                # prepare image for visualization
                sample = vis_transform(**sample)
                img = sample['image'].copy()

                # prepare sample for model applying
                sample = to_tensor_transform(**sample)
                sample['image'] = sample['image'].to(device)

                # applying model
                output = model([sample['image']])
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
                    model_save_folder = os.path.join(
                        args.save_to,
                        os.path.basename(model_path)[:-4] + '_preds'
                    )
                    os.makedirs(model_save_folder, exist_ok=True)
                    save_path = os.path.join(model_save_folder,
                                             os.path.basename(img_path))
                    cv2.imwrite(save_path, drawn_img)

        # calculate metrics if markup file was passed
        if markup:
            print('\nMetrics for model:', os.path.basename(model_path))
            metrics = evaluator.calculate_metrics()
            for name, value in metrics.items():
                print(f'{name}: {value}')


if __name__ == '__main__':
    main()
