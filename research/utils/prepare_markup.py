#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import json
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
                        help='Path to dir for save results.')
    parser.add_argument('--train-ratio', type=float,
                        help='Value in range [0, 1] for split given dataset '
                             'into train and test.')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize markup.')
    parser.add_argument('--fix-markup', action='store_true',
                        help='Search and remove overlapped bounding boxes.')
    parser.add_argument('--iou-th', type=float, default=0.65,
                        help='IoU threshold for bbox filtering.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print additional information.')

    args = parser.parse_args()

    check_or_create_dir(args.save_to, critical=True)

    if not (0 < args.iou_th <= 1.0):
        print('Bad value for "--iou-th": it must be in range (0; 1].')
        exit(1)

    return args


def calc_iou_bbox(box_1, box_2):
    """Calculate intersection over union between two bounding boxes.

    Parameters
    ----------
    box_1 : array-like
        Array with four 2D points.
    box_2 : array-like
        Array with four 2D points.

    Returns
    -------
    float
        Calculated IoU.

    """
    x_a = max(box_1[0], box_2[0])
    y_a = max(box_1[1], box_2[1])
    x_b = min(box_1[2], box_2[2])
    y_b = min(box_1[3], box_2[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box_b_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    return inter_area / float(box_a_area + box_b_area - inter_area)


def filter_bboxes_by_iou(bboxes, labels, iou_th=0.65, verbose=False):
    """Filter bboxes if some of them intersecting more than iou_th.

    Parameters
    ----------
    bboxes : array-like
        Array with 2D bounding boxes (4 points).
    labels : array-like
        Labels for bounding boxes.
    iou_th : float
        IoU threshold for filtering.
    verbose : bool
        Print additional information.

    Returns
    -------
    array-like
        Filterred bounding boxes.

    """
    thresholded_idxs = set()
    new_bboxes, new_labels = [], []
    ious = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            iou = calc_iou_bbox(bboxes[i], bboxes[j])
            if iou != 0:
                ious.append(iou)

            if iou > iou_th:
                # remove negative(normal) bbox
                if labels[i]:
                    thresholded_idxs.add(j)
                else:
                    thresholded_idxs.add(i)
                    break

        if i not in thresholded_idxs:
            new_bboxes.append(bboxes[i])
            new_labels.append(labels[i])

    if verbose:
        if len(ious) > 0:
            ious = np.array(ious)
            print('\nMin IoU:', ious.min())
            print('Max IoU:', ious.max())
            print('Mean Iou', ious.mean())
            print('Median IoU', np.median(ious))
            print('Thresholded pairs:', len(thresholded_idxs) // 2)
        else:
            print('\nNo intersected polygons.')

    return new_bboxes, new_labels


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
                    if args.verbose:
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
                cur_bb = []
                for obj in objs:
                    # store label
                    if obj['name'] == zero_label:
                        cur_labels.append(False)
                    else:
                        cur_labels.append(True)

                    # process bbox
                    polygon = np.array(
                        [[int(p['x']), int(p['y'])]
                         for p in obj['polygon']['pt']]
                    )

                    # check that bbox is rect
                    set_x = set(polygon[:, 0])
                    set_y = set(polygon[:, 1])
                    if len(set_x) > 2 or len(set_y) > 2:
                        print('\nWARNING: bad bounding box:', polygon)
                        print('For sample:', df['Файлы'][idx])
                        continue

                    bb = [min(set_x), min(set_y), max(set_x), max(set_y)]
                    # cast to int for right json serialization
                    cur_bb.append([int(x) for x in bb])

                labels.append(cur_labels)
                bboxes.append(cur_bb)

    if args.verbose:
        print('\nTotal samples:', total_samples)
        print('Good samples:', len(img_paths))

    if args.fix_markup:
        for i in range(len(bboxes)):
            bboxes[i], labels[i] = filter_bboxes_by_iou(
                bboxes[i],
                labels[i],
                args.iou_th,
                args.verbose
            )

    # save prepared markup
    with open(os.path.join(args.save_to, 'markup.json'), 'w') as f:
        markup = []
        for path, cur_lb, cur_bb in zip(img_paths, labels, bboxes):
            ant = [{'label': lb, 'bbox': bb} for lb, bb in zip(cur_lb, cur_bb)]
            markup.append({'img_path': path, 'annotation': ant})

        json.dump(markup, fp=f, indent=4)

    if args.visualize:
        vis_images = os.path.join(args.save_to, 'vis_images')
        if not os.path.exists(vis_images):
            os.mkdir(vis_images)

        print('\nVisualizing:')
        for img_path, cur_lbs, cur_bb in tqdm(zip(img_paths, labels, bboxes)):
            img = cv2.imread(img_path)

            # drawing bboxes
            for label, bb in zip(cur_lbs, cur_bb):
                pts = np.array([[bb[0], bb[1]],
                                [bb[0], bb[3]],
                                [bb[2], bb[3]],
                                [bb[2], bb[1]]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = (0, 0, 255) if label else (0, 255, 0)
                img = cv2.polylines(img, [pts], True, color)

            cv2.imwrite(
                os.path.join(vis_images, os.path.basename(img_path)),
                img
            )


if __name__ == '__main__':
    main()
