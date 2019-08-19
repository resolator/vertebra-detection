#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score

sys.path.append(os.path.join(sys.path[0], '../common'))
from utils import calc_iou_bbox


def postprocessing(outputs, iou_th=0.65):
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


def calc_metrics(outputs, targets, iou_th=0.5):
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

    if len(all_gt_labels) == 0 and len(all_pd_labels) == 0:
        return np.zeros(5)

    return np.array([
        precision_score(all_gt_labels, all_pd_labels, pos_label=2),
        recall_score(all_gt_labels, all_pd_labels, pos_label=2),
        f1_score(all_gt_labels, all_pd_labels, pos_label=2),
        average_precision_score(all_gt_labels, all_pd_labels, pos_label=2),
        np.mean(collected_ious),
        matched_boxes_count / boxes_count
    ])
