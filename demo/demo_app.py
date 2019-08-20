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

import torch
from torchvision import transforms, models

sys.path.append(os.path.join(sys.path[0], '../common'))
from utils import postprocessing, match_labels


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

    args = parser.parse_args()
    if args.save_to is not None:
        os.makedirs(args.save_to, exist_ok=True)

    return args


def bb2pts(bb):
    """Convert bounding box to cv2 pts."""
    pts = np.array([[bb[0], bb[1]],
                    [bb[0], bb[3]],
                    [bb[2], bb[3]],
                    [bb[2], bb[1]]], np.int32)

    return pts.reshape((-1, 1, 2))


def draw_bboxes(img, bboxes, labels):
    """Draw bounding box on image and return drawn image."""
    for box, label in zip(bboxes, labels):
        pts = bb2pts(box)
        color = (0, 0, 255) if label else (0, 255, 0)
        img = cv2.polylines(img, [pts], True, color)

    return img


def main():
    """Application entry point."""
    args = get_args()

    if os.path.isfile(args.images):
        with open(args.images) as f:
            markup_file = json.load(f)
            images_paths = [x['img_path'] for x in markup_file]

    else:
        images_paths = os.listdir(args.images)

    # model preparation
    ckpt = torch.load(args.model_path)

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                     num_classes=2)
    model.load_state_dict(ckpt['model'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ckpt['args'].mean, ckpt['args'].std)
    ])

    cpu_device = torch.device('cpu')
    with torch.no_grad():
        for img_path in tqdm(images_paths, desc='Predicting'):
            img = Image.open(img_path)
            prepared_img = transform(img)

            output = model([prepared_img])
            output = {k: v.to(cpu_device).numpy()
                      for k, v in output[0].items()}

            img = cv2.imread(img_path)
            drawn_img = draw_bboxes(img, output['boxes'], output['labels'])

            if args.save_to is None:
                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.imshow('img', drawn_img)
                cv2.waitKey()

            else:
                save_path = os.path.join(args.save_to,
                                         os.path.basename(img_path))
                cv2.imwrite(save_path, drawn_img)


if __name__ == '__main__':
    main()
