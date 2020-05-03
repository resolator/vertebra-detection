#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converts the model to torchscript.
"""
import torch
import argparse

from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path to model.pth.')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to zip file with saved model.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()
    device = torch.device('cpu')

    # load model from ckpt
    ckpt = torch.load(str(args.model_path), map_location=device)
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # add additional params into model
    model.resize_size = ckpt['args'].resize_size
    model.center_crop_size = ckpt['args'].center_crop_size
    model.min_visibility = ckpt['args'].min_visibility
    model.mean = ckpt['args'].mean
    model.std = ckpt['args'].std

    # pack and save
    scripted_model = torch.jit.script(model)
    scripted_model.save(str(args.save_to))
    print('Model saved successfully.')


if __name__ == '__main__':
    main()
