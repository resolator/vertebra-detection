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
from utils import calc_iou_bbox, draw_bboxes


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(
        description='Script for markup preparation.'
    )
    parser.add_argument('--config', is_config_file=True,
                        help='Path to config file.')
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
    parser.add_argument('--markup-format', choices=['default', 'yolov3'],
                        default='default',
                        help='Format of resulted markup.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print additional information.')
    parser.add_argument('--calc-mean-std', action='store_true',
                        help='Calculate mean and std on given images.')

    args = parser.parse_args()

    os.makedirs(args.save_to, exist_ok=True)

    if not (0 < args.iou_th <= 1):
        print('Bad value for "--iou-th": it must be in range (0; 1].')
        exit(1)

    if args.train_ratio is not None:
        if not (0 < args.train_ratio < 1):
            print('Bad value for "--train-ratio": it must be in range (0; 1).')
            exit(1)

    return args


def split_by_ratio(array, ratio):
    """Randomly split array in two parts.

    Parameters
    ----------
    array : array-like
        Array for splitting.
    ratio : float
        Splitting ratio in range (0; 1).

    Returns
    -------
    tuple
        Two randomly splitted arrays.

    """
    indices = np.arange(len(array))
    np.random.shuffle(indices)

    size = int(len(array) * ratio)
    return array[:size], array[size:]


def convert_bbox_to_yolov3(bbox, img_size):
    """Convert coordinates to yolov3 markup format.

    Parameters
    ----------
    bbox : array-like
        Array with 4 ints [x_min, y_min, x_max, y_max].
    img_size : dict
        Dict with image width and height.

    Returns
    -------
    List
        Converted bounding box data.

    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x_center = bbox[0] + width / 2  # center by width
    y_center = bbox[1] + height / 2  # center by height

    # normalize
    x_center /= img_size['width']
    y_center /= img_size['height']
    width /= img_size['width']
    height /= img_size['height']

    ints = [x_center, y_center, width, height]
    # cast to str for right serialization
    return [str(x) for x in ints]


def save_markup_as_yolov3(img_paths,
                          labels,
                          bboxes,
                          sizes,
                          save_to,
                          train_ratio):
    """Prepare and save markup for YOLOv3.

    Parameters
    ----------
    img_paths : array-like
        Array with images paths.
    labels : array-like
        Array with arrays of labels for every image.
    bboxes : array-like
        Array with arrays of bounding boxes for every image.
    sizes : array-like
        Array with size for every image.
    save_to : str
        Part to saving folder.
    train_ratio : float
        Splitting ratio in range (0; 1).

    """
    # split in train/test
    splitted_paths = split_by_ratio(img_paths, train_ratio)

    # save images paths
    for phase, paths in zip(['train', 'test'], splitted_paths):
        with open(os.path.join(save_to, phase + '.txt'), 'w') as f:
            [print(path, file=f) for path in paths]

    # save classes.name
    classes_path = os.path.join(save_to, 'classes.name')
    with open(classes_path, 'w') as f:
        print('normal', file=f)
        print('anomaly', file=f)

    # save main description file
    train_path = os.path.join(save_to, 'train.txt')
    test_path = os.path.join(save_to, 'test.txt')
    labels_dir = os.path.join(save_to, 'labels')
    with open(os.path.join(save_to, 'custom.data'), 'w') as f:
        print('classes= 2', file=f)
        print('train=' + train_path, file=f)
        print('valid=' + test_path, file=f)
        print('names=' + classes_path, file=f)
        print('labels=' + labels_dir, file=f)

    # process and save labels
    os.makedirs(labels_dir, exist_ok=True)
    for path, cur_lbs, cur_bbs, size in zip(img_paths, labels, bboxes, sizes):
        # change extention to .txt
        cur_name = '.'.join(os.path.basename(path).split('.')[:-1] + ['txt'])

        # process and save labels and bboxes for current image name
        with open(os.path.join(labels_dir, cur_name), 'w') as f:
            for label, bbox in zip(cur_lbs, cur_bbs):
                converted_bbox = convert_bbox_to_yolov3(bbox, size)
                # prepare text line
                line = ' '.join([str(int(label))] + converted_bbox)
                print(line, file=f)


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


def calc_mean_std(images_dir):
    r_px = []
    g_px = []
    b_px = []
    gray_px = []
    for name in tqdm(os.listdir(images_dir), desc='Collecting pixels'):
        img_path = os.path.join(images_dir, name)

        img = cv2.imread(img_path)
        r_px += img[:, :, 2].ravel().tolist()
        g_px += img[:, :, 1].ravel().tolist()
        b_px += img[:, :, 0].ravel().tolist()

        img = cv2.imread(img_path, 0)
        gray_px += img.ravel().tolist()

    print('Processing pixels...')
    r_px = np.array(r_px)
    g_px = np.array(g_px)
    b_px = np.array(b_px)
    gray_px = np.array(gray_px)

    for arr, name in zip([r_px, g_px, b_px, gray_px],
                         ['Red', 'Green', 'Blue', 'Gray']):
        print('\n{}:\nMean: {}\nSTD: {}'.format(
            name, arr.mean() / 255, arr.std() / 255
        ))


def main():
    """Application entry point."""
    args = get_args()

    good_for_markup_key = 'На срезе визуализируются межпозвоночные диски'
    good_for_markup_value = 'Визуализируются (можно размечать)'
    zero_label = 'shejnyj-mezhpozvonochnyj-disk-zdorovyj'

    img_paths, labels, bboxes, sizes = [], [], [], []
    total_samples = 0

    # collect and parse source markup
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
                    annot = xml['annotation']
                else:
                    annot = xml['annotationgroup']['annotation']

                sizes.append({'height': int(annot['imagesize']['nrows']),
                              'width': int(annot['imagesize']['ncols'])})

                cur_labels = []
                cur_bb = []
                for obj in annot['object']:
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
    markup = []
    if args.markup_format == 'yolov3':
        save_markup_as_yolov3(
            img_paths, labels, bboxes, sizes, args.save_to, args.train_ratio
        )
    else:
        for path, cur_lb, cur_bb in zip(img_paths, labels, bboxes):
            annot = [{'label': lb, 'bbox': bb}
                     for lb, bb in zip(cur_lb, cur_bb)]
            markup.append({'img_path': path, 'annotation': annot})

        if args.train_ratio is None:
            with open(os.path.join(args.save_to, 'markup.json'), 'w') as f:
                json.dump(markup, fp=f, indent=4)
        else:
            splitted_markup = split_by_ratio(markup, args.train_ratio)
            for ph, mkp in zip(['train', 'test'], splitted_markup):
                with open(os.path.join(args.save_to, ph + '.json'), 'w') as f:
                    json.dump(mkp, fp=f, indent=4)

    if args.calc_mean_std:
        calc_mean_std(args.images_dir)

    # visualization
    if args.visualize:
        vis_images = os.path.join(args.save_to, 'vis_images')
        os.makedirs(vis_images, exist_ok=True)

        print('\nVisualizing:')
        for img_path, cur_lbs, cur_bb in tqdm(zip(img_paths, labels, bboxes)):
            img = cv2.imread(img_path)
            img = draw_bboxes(img, cur_bb, cur_lbs)

            cv2.imwrite(
                os.path.join(vis_images, os.path.basename(img_path)),
                img
            )


if __name__ == '__main__':
    main()
