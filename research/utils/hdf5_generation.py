#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import configargparse

import pandas as pd

from xml.etree import ElementTree as ET

sys.path.append(os.path.join(sys.path[0], '../common'))
from utils import check_or_create_dir


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(
        description='Script for hdf5-files generation with dataset.'
    )
    parser.add_argument('--images-dir', required=True,
                        help='Path to dir with images.')
    parser.add_argument('--markup-dir', required=True,
                        help='Path to dir with markup csv files.')
    parser.add_argument('--save-to', required=True,
                        help='Path to dir for save resulted datasets.')
    parser.add_argument('--train-ratio', type=float,
                        help='Value in range [0, 1] for split given dataset '
                             'into train and test.')

    args = parser.parse_args()

    check_or_create_dir(args.save_to, critical=True)

    return args


def markup_parser():
    pass


def main():
    """Application entry point."""
    args = get_args()

    markup = []
    for mkp_name in os.listdir(args.markup_dir):
        mkp_path = os.path.join(args.markup_dir, mkp_name)
        df = pd.read_csv(mkp_path)

        polygons = []
        for sample_xml in df['XML']:
            sample = ET.fromstring(sample_xml)
            for element in list(sample):
                if element.tag == 'object':
                    pass



if __name__ == '__main__':
    main()
