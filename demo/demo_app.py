#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from math import isnan

import torch
from torchvision import transforms, models

sys.path.append(os.path.join(sys.path[0], '../'))
from common.utils import postprocessing, draw_bboxes
from common.metrics import calc_metrics


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
    if os.path.isfile(args.images):
        with open(args.images) as f:
            samples = json.load(f)
            markup = True
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
        ckpt = torch.load(model_path)

        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                         num_classes=3)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(ckpt['args'].mean, ckpt['args'].std)
        ])

        m_names = ['Precision', 'Recall', 'F1', 'mAP']
        metrics = [[], [], [], []]
        cpu_device = torch.device('cpu')
        with torch.no_grad():
            for sample in tqdm(samples, desc='Predicting'):
                # get and process output
                if markup:
                    img_path = sample['img_path']
                else:
                    img_path = sample

                img = Image.open(img_path)
                prepared_img = transform(img).to(device)

                output = model([prepared_img])
                output = {k: v.to(cpu_device).numpy()
                          for k, v in output[0].items()}
                output = postprocessing([output], iou_th=args.iou_th)[0]

                img = cv2.imread(img_path)
                drawn_img = draw_bboxes(img, output['boxes'], output['labels'],
                                        shifted_labels=True)

                # evaluate if markup file was passed
                if markup:
                    target = {
                        'boxes': [x['bbox'] for x in sample['annotation']],
                        'labels': [int(x['label'])
                                   for x in sample['annotation']]
                    }
                    cur_m = np.array(calc_metrics(output, target))
                    [m.append(x)
                     for x, m in zip(cur_m, metrics) if not isnan(x)]

                    img = cv2.imread(img_path)
                    gt_drawn_img = draw_bboxes(
                        img,
                        target['boxes'],
                        target['labels'],
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
            for m_name, m in zip(m_names, metrics):
                print(f'{m_name}: {np.mean(m)}')


if __name__ == '__main__':
    main()
