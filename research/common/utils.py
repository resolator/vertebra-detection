#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch


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