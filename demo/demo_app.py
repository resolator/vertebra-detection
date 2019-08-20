#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import argparse
import sklearn.metrics as sk_metrics
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms, models

sys.path.append(os.path.join(sys.path[0], '../'))
from common.utils import postprocessing, match_labels, draw_bboxes


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(
        description='Demo application for vertebra classification.'
    )
    parser.add_argument('--images', required=True,
                        help='Path to dir with images for classification. '
                             'It also can be a markup file for evaluation.')
    parser.add_argument('--model-path', default='../data/model.pth',
                        help='Path to trained model.')
    parser.add_argument('--save-to',
                        help='Path to dir for store classification results '
                             'instead of displaying them.')
    parser.add_argument('--iou-th', type=float, default=0.65,
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

    # model preparation
    ckpt = torch.load(args.model_path)

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                     num_classes=3)
    model.load_state_dict(ckpt['model'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ckpt['args'].mean, ckpt['args'].std)
    ])

    all_gt_labels, all_pd_labels = [], []
    cpu_device = torch.device('cpu')
    with torch.no_grad():
        for sample in tqdm(samples, desc='Predicting'):
            # get and process output
            if markup:
                img_path = sample['img_path']
            else:
                img_path = sample

            img = Image.open(img_path)
            prepared_img = transform(img)

            output = model([prepared_img])
            output = {k: v.to(cpu_device).numpy()
                      for k, v in output[0].items()}
            output = postprocessing([output], iou_th=args.iou_th)[0]

            # evaluate if markup file was passed
            if markup is not None:
                target = [{
                    'boxes': [x['bbox'] for x in sample['annotation']],
                    'labels': [int(x['label']) for x in sample['annotation']]
                }]
                gt_labels, pd_labels = match_labels([output], target)
                all_gt_labels += gt_labels
                all_pd_labels += pd_labels

            img = cv2.imread(img_path)
            drawn_img = draw_bboxes(img, output['boxes'], output['labels'],
                                    shifted_labels=True)

            if args.save_to is None:
                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.imshow('img', drawn_img)
                cv2.waitKey()

            else:
                save_path = os.path.join(args.save_to,
                                         os.path.basename(img_path))
                cv2.imwrite(save_path, drawn_img)

    # calculate metrics if markup file was passed
    if len(all_gt_labels) > 0:
        print()
        m_names = ['Precision', 'Recall', 'F1']
        m_funcs = [sk_metrics.precision_score,
                   sk_metrics.recall_score,
                   sk_metrics.f1_score]

        for m_name, m_func in zip(m_names, m_funcs):
            m = m_func(all_gt_labels, all_pd_labels, pos_label=2)
            print(f'{m_name}: {m}')


if __name__ == '__main__':
    main()
