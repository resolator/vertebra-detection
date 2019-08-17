#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.join(sys.path[0], '../common'))
from utils import calc_iou_bbox


def calc_metrics(outputs, targets, iou_th=0.5):
    collected_gt_labels = []
    collected_pd_labels = []
    collected_ious = []

    for gt, pd in zip(targets, outputs):
        sorted_gt_boxes = []
        sorted_pd_boxes = []

        sorted_gt_labels = []
        sorted_pd_labels = []

        pd_skip_indices = []
        sorted_ious = []

        for i, gt_box in enumerate(gt['boxes']):
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
                    sorted_ious.append(ious[max_idx])

                    sorted_gt_boxes.append(gt['boxes'][i])
                    sorted_pd_boxes.append(pd['boxes'][max_idx])

                    sorted_gt_labels.append(gt['labels'][i])
                    sorted_pd_labels.append(pd['labels'][max_idx])

                    pd_skip_indices.append(max_idx)

        collected_gt_labels += sorted_gt_labels
        collected_pd_labels += sorted_pd_labels
        collected_ious += sorted_ious

    return np.array([precision_score(collected_gt_labels, collected_pd_labels),
                     recall_score(collected_gt_labels, collected_pd_labels),
                     f1_score(collected_gt_labels, collected_pd_labels),
                     np.mean(collected_ious)])