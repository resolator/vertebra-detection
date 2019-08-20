#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

sys.path.append(os.path.join(sys.path[0], '../'))
from common.utils import calc_iou_bbox


def postprocessing(outputs, iou_th=0.65):
    """Post processing for FasterRCNN output.

    Parameters
    ----------
    outputs : List
        Outputs from FasterRCNN in evaluation mode.
    iou_th : float
        Threshold for remove very intersecting boxes.

    Returns
    -------
    List
        Processed outputs.

    """
    resulted_outputs = []
    for output in outputs:
        removed_indices = []
        for i in range(len(output['boxes'])):
            for j in range(i, len(output['boxes'])):
                if i == j:
                    continue

                iou = calc_iou_bbox(output['boxes'][i], output['boxes'][j])
                # remove box with lower score
                if iou > iou_th:
                    if output['scores'][i] > output['scores'][j]:
                        removed_indices.append(j)
                    else:
                        removed_indices.append(i)

        boxes, labels, scores = [], [], []
        for i in range(len(output['boxes'])):
            if i not in removed_indices:
                boxes.append(output['boxes'][i])
                labels.append(output['labels'][i])
                scores.append(output['scores'][i])

        resulted_outputs.append(
            {'boxes': boxes, 'labels': labels, 'scores': scores}
        )

    return resulted_outputs


def match_labels(outputs, targets, iou_th=0.5):
    """Match GT labels to PD labels using boxes intersection.

    Parameters
    ----------
    outputs : List
        Outputs from FasterRCNN in evaluation mode.
    targets : List
        Same as outputs but ground truth.
    iou_th : float
        Threshold for match GT and PD boxes.

    Returns
    -------
    tuple
        Tuple with matched_gt_labels and matched_pd_labels.

    """
    all_gt_labels = []
    all_pd_labels = []
    collected_ious = []

    boxes_count = 0
    matched_boxes_count = 0

    for gt, pd in zip(targets, outputs):
        sorted_gt_boxes = []
        sorted_pd_boxes = []

        sorted_gt_labels = []
        sorted_pd_labels = []

        pd_skip_indices = []
        sorted_ious = []

        for i, gt_box in enumerate(gt['boxes']):
            boxes_count += 1
            if len(pd_skip_indices) == len(pd['boxes']):
                break

            ious = []
            for j, pd_box in enumerate(pd['boxes']):
                if j in pd_skip_indices:
                    ious.append(0)
                    continue
                ious.append(calc_iou_bbox(gt_box, pd_box))

            if len(ious) > 0:
                max_idx = np.argmax(ious)
                if ious[max_idx] >= iou_th:
                    matched_boxes_count += 1
                    sorted_ious.append(ious[max_idx])

                    sorted_gt_boxes.append(gt['boxes'][i])
                    sorted_pd_boxes.append(pd['boxes'][max_idx])

                    sorted_gt_labels.append(gt['labels'][i])
                    sorted_pd_labels.append(pd['labels'][max_idx])

                    pd_skip_indices.append(max_idx)

        all_gt_labels += sorted_gt_labels
        all_pd_labels += sorted_pd_labels
        collected_ious += sorted_ious

    return all_gt_labels, all_pd_labels
