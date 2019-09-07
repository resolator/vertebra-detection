#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import torch

import numpy as np


def bb2pts(bb):
    """Convert bounding box to cv2 pts."""
    pts = np.array([[bb[0], bb[1]],
                    [bb[0], bb[3]],
                    [bb[2], bb[3]],
                    [bb[2], bb[1]]], np.int32)

    return pts.reshape((-1, 1, 2))


def draw_bboxes(img, boxes, labels, shifted_labels=False, mean=None, std=None):
    """Draw bounding box at the image.

    Parameters
    ----------
    img : numpy array or tensor
        Image for drawing.
    boxes : List
        Boxes for drawing
    labels : List
        Labels for each box.
    shifted_labels : bool
        Set it True if negative label = 1 and positive label = 2.
    std : List
        Standart deviation for every channel (RGB order). You have to pass it
        if the img is a tensor.
    mean : List
        Mean for every channel (RGB order). You have to pass it if the img is
        a tensor.

    Returns
    -------
    numpy array
        Drawn img.

    """
    from_tensor = (mean is not None) and (std is not None)
    if from_tensor:
        img = img.clone().detach()
        # denormalize
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)

        # convert to cv2 format
        img = img.cpu().numpy().transpose((1, 2, 0))

        # convert to uint8 for drawing
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    for box, label in zip(boxes, labels):
        if shifted_labels:
            if from_tensor:
                color = (255, 0, 0) if label == 2 else (0, 255, 0)
            else:
                    color = (0, 0, 255) if label == 2 else (0, 255, 0)
        else:
            color = (0, 0, 255) if label else (0, 255, 0)

        pts = bb2pts(box)
        img = cv2.polylines(img, [pts], True, color)

    if from_tensor:
        if isinstance(img, cv2.UMat):
            img = img.get()
        return (img / 255).astype(np.float32)

    return img


def calc_iou_bbox(box_1, box_2, tensors=False):
    """Calculate intersection over union between two bounding boxes.

    Parameters
    ----------
    box_1 : array-like
        Array with four 2D points.
    box_2 : array-like
        Array with four 2D points.
    tensors : bool
        If true then calculate on via torch.

    Returns
    -------
    float
        Calculated IoU.

    """
    if tensors:
        x_a = torch.max(torch.tensor([box_1[0], box_2[0]], dtype=torch.float))
        y_a = torch.max(torch.tensor([box_1[1], box_2[1]], dtype=torch.float))
        x_b = torch.min(torch.tensor([box_1[2], box_2[2]], dtype=torch.float))
        y_b = torch.min(torch.tensor([box_1[3], box_2[3]], dtype=torch.float))

        inter_area = torch.max(torch.tensor([0, x_b - x_a + 1])) * \
                     torch.max(torch.tensor([0, y_b - y_a + 1]))

        box_a_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
        box_b_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

        return inter_area / (box_a_area + box_b_area - inter_area)

    x_a = max(box_1[0], box_2[0])
    y_a = max(box_1[1], box_2[1])
    x_b = min(box_1[2], box_2[2])
    y_b = min(box_1[3], box_2[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box_b_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    return inter_area / float(box_a_area + box_b_area - inter_area)


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
            total_iou = 0
            for j in range(i, len(output['boxes'])):
                if i == j:
                    continue

                total_iou += calc_iou_bbox(output['boxes'][i], output['boxes'][j])
                # remove box with lower score
                if total_iou > iou_th:
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
