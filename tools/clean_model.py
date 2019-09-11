#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import argparse

from tqdm import tqdm


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(
        description='Script for clean model.'
    )
    parser.add_argument('--model-path', required=True,
                        help='Path to model or dir with models.')
    parser.add_argument('--save-to', required=True,
                        help='Path to saving dir.')

    args = parser.parse_args()
    os.makedirs(args.save_to, exist_ok=True)

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if os.path.isfile(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        save_name = os.path.join(args.save_to,
                                 os.path.basename(args.model_path))
        torch.save({'model': ckpt['model'],
                    'args': ckpt['args']}, save_name)
    else:
        for model_name in tqdm(os.listdir(args.model_path), desc='Cleaning'):
            model_path = os.path.join(args.model_path, model_name)
            ckpt = torch.load(model_path, map_location=device)
            save_name = os.path.join(args.save_to, model_name)
            torch.save({'model': ckpt['model'],
                        'args': ckpt['args']}, save_name)


if __name__ == '__main__':
    main()
