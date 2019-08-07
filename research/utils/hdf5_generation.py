#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import configargparse

sys.path.append(os.path.join(sys.path[0], '../common'))
from utils import check_or_create_dir


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(
        description='Script for hdf5-files generation with dataset.'
    )
    parser.add_argument('--data-dir', required=True,
                        help='Path to dir with raw data.')
    parser.add_argument('--save-to', required=True,
                        help='Path to dir for save resulted datasets.')
    parser.add_argument('--train-ratio', type=float,
                        help='Value in range [0, 1] for split given dataset '
                             'into train and test.')

    args = parser.parse_args()

    check_or_create_dir(args.save_to, critical=True)

    return args


def main():
    """Application entry point."""
    args = get_args()


if __name__ == '__main__':
    main()
