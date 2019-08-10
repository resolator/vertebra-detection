#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import xmltodict
import configargparse

import numpy as np
import pandas as pd

from tqdm import tqdm

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
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize markup.')

    args = parser.parse_args()

    check_or_create_dir(args.save_to, critical=True)

    return args


def main():
    """Application entry point."""
    args = get_args()

    good_for_markup_key = 'На срезе визуализируются межпозвоночные диски'
    good_for_markup_value = 'Визуализируются (можно размечать)'
    zero_label = 'shejnyj-mezhpozvonochnyj-disk-zdorovyj'

    img_paths, labels, bboxes = [], [], []
    total_samples = 0

    for mkp_name in os.listdir(args.markup_dir):
        mkp_path = os.path.join(args.markup_dir, mkp_name)
        df = pd.read_csv(mkp_path)

        for idx, rec in enumerate(df[good_for_markup_key]):
            total_samples += 1
            if rec == good_for_markup_value:
                # decoding annotation
                try:
                    xml = xmltodict.parse(df['XML'][idx])
                except Exception as e:
                    print('\nBroken XML:', df['XML'][idx])
                    print('For image:', df['Файлы'][idx])
                    print('Error:', e)
                    continue

                # make and store img_path
                img_paths.append(os.path.join(
                    args.images_dir,
                    df['Файлы'][idx].replace('/n', '')
                ))

                # extract labels and bboxes
                if 'annotation' in xml.keys():
                    objs = xml['annotation']['object']
                else:
                    objs = xml['annotationgroup']['annotation']['object']

                cur_labels = []
                cur_bboxes = []
                for obj in objs:
                    # store label
                    if obj['name'] == zero_label:
                        cur_labels.append(False)
                    else:
                        cur_labels.append(True)

                    # store bbox
                    cur_bboxes.append(
                        [[p['x'], p['y']] for p in obj['polygon']['pt']]
                    )

                labels.append(cur_labels)
                bboxes.append(cur_bboxes)

    print('\nTotal samples:', total_samples)
    print('Good samples:', len(img_paths))

    if args.visualize:
        print('\nVisualizing:')
        for img_path, cur_lbs, cur_bb in tqdm(zip(img_paths, labels, bboxes)):
            img = cv2.imread(img_path)

            for label, bbox in zip(cur_lbs, cur_bb):
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = (0, 255, 0) if label else (0, 0, 255)
                img = cv2.polylines(img, [pts], True, color)

            cv2.imwrite(
                os.path.join(args.save_to, os.path.basename(img_path)),
                img
            )


if __name__ == '__main__':
    main()
