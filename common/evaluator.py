#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import precision_recall_curve
from .utils import calc_iou_bbox


class Evaluator(object):
    """Evaluator for the FasterRCNN.

    Use it for collect stats for every iteration in the epoch and calculate
    metrics at the end of the epoch.

    Parameters
    ----------
    iou_th : float
        Threshold for match GT and PD boxes.

    """
    def __init__(self, iou_th=0.5):
        self.iou_th = iou_th
        self.metrics_names = ('precision', 'recall', 'f1', 'aver_prec')

        self._storage = None
        self._init_storage()

    def _init_storage(self):
        """Clean stored stats."""
        self._storage = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'y_true': [],
            'y_score': []
        }

    def collect_stats(self, outputs, targets):
        """"Calculate and collect stats for metrics calculation.

        Parameters
        ----------
        outputs : List
            Outputs from FasterRCNN in evaluation mode.
        targets : List
            Same as outputs but ground truth.

        """
        for output, target in zip(outputs, targets):
            self._store_tfpn(output, target)

    def calculate_metrics(self):
        """Calculate metrics on collected stats and reset storage of stats.

        Returns
        -------
        dict
            Calculated metrics.

        """
        if len(self._storage['y_true']) == 0:
            return 0, 0, 0, 0

        # precision calculation
        prec_div = self._storage['tp'] + self._storage['fp']
        if prec_div != 0:
            precision = self._storage['tp'] / prec_div
        else:
            precision = 0

        # recall calculation
        rec_div = self._storage['tp'] + self._storage['fn']
        if rec_div != 0:
            recall = self._storage['tp'] / rec_div
        else:
            recall = 0

        # f1 calculation
        f1_div = precision + recall
        if f1_div != 0:
            f1 = 2 * precision * recall / f1_div
        else:
            f1 = 0

        # average precision calculation
        p, r, _ = precision_recall_curve(
            self._storage['y_true'], self._storage['y_score'], pos_label=2
        )
        aver_prec = -np.sum(np.diff(r) * np.array(p)[:-1])

        # clean stored stats
        self._init_storage()

        return {'precision': precision,
                'recall': recall,
                'f1': f1,
                'aver_prec': aver_prec}

    def _store_tfpn(self, output, target):
        """Extract information for metrics calculation.

        Parameters
        ----------
        output : dict
            Outputs from FasterRCNN in evaluation mode.
        target : dict
            Same as outputs but ground truth.

        """
        tp, fp, fn = 0, 0, 0
        y_true, y_score = [], []
        for i, gt_box in enumerate(target['boxes']):
            tp_found = False
            for j, pd_box in enumerate(output['boxes']):
                if calc_iou_bbox(gt_box, pd_box) >= self.iou_th:
                    y_true.append(target['labels'][i])
                    y_score.append(output['scores'][j])
                    if target['labels'][i] == output['labels'][j]:
                        if not tp_found:
                            tp_found = True
                            tp += 1
                    else:
                        fn += 1

            fp = len(target['boxes']) - tp

        self._storage['tp'] += tp
        self._storage['fp'] += fp
        self._storage['fn'] += fn
        self._storage['y_true'] += y_true
        self._storage['y_score'] += y_score
