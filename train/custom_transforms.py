#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import torch
from torchvision.transforms import functional as F


class Resize(object):
    """Wrapper for processing custom input.

    Input must be presented as dict with the following keys: 'img', 'bboxes'.

    Attributes
    ----------
    width : int
        Target width.
    height : int
        Target height.

    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _bboxes_resize(self, img_size, bboxes):
        w_factor = self.width / img_size[0]
        h_factor = self.height / img_size[1]

        return [[int(bb[0] * w_factor),
                 int(bb[1] * h_factor),
                 int(bb[2] * w_factor),
                 int(bb[3] * h_factor)] for bb in bboxes]

    def __call__(self, sample):
        sample['bboxes'] = self._bboxes_resize(
            sample['img'].size, sample['bboxes']
        )
        sample['img'] = F.resize(sample['img'], (self.height, self.width))

        return sample


class Crop(object):
    """Wrapper for processing custom input.

    Input must be presented as dict with the following keys: 'img', 'bboxes'.

    Attributes
    ----------
    crop_factor : float
        Factor for determinate the target size (how much width and height we
        should remove) Must be in range (0; 1).
    center_crop : bool
        Use center crop instead of bottom-right crop.

    """
    def __init__(self, crop_factor, center_crop=False):
        if 0 < crop_factor < 1:
            raise AttributeError('"crop_factor" must be in range (0; 1)!')

        self.crop_factor = crop_factor
        self.center_crop = center_crop

    def __call__(self, sample):
        x = self.crop_factor * sample['img'].size[0]
        y = self.crop_factor * sample['img'].size[1]
        if self.center_crop:
            w = sample['img'].size[0] - x * 2
            h = sample['img'].size[1] - y * 2
        else:
            w = sample['img'].size[0] - x
            h = sample['img'].size[1] - y

        sample['img'] = F.crop(sample['img'], x, y, h, w)
        sample['bboxes'] = [[round(bb[0] - x),
                             round(bb[1] - y),
                             round(bb[2] - x),
                             round(bb[3] - y)] for bb in sample['bboxes']]

        return sample


class RandomHorizontalFlip(object):
    """Wrapper for processing custom input.

    Flip image with their bounding box.
    Input must be presented as dict with the following keys: 'img', 'bboxes'.

    Attributes
    ----------
    prob : float
        Probability of flip.

    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['img'] = F.hflip(sample['img'])
            h_size = sample['img'].size[1]
            sample['bboxes'] = [[h_size - bb[2],
                                 bb[1],
                                 h_size - bb[0],
                                 bb[3]] for bb in sample['bboxes']]

        return sample


class RandomVerticalFlip(object):
    """Wrapper for processing custom input.

    Flip image with their bounding box.
    Input must be presented as dict with the following keys: 'img', 'bboxes'.

    Attributes
    ----------
    prob : float
        Probability of flip.

    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['img'] = F.vflip(sample['img'])
            w_size = sample['img'].size[0]
            sample['bboxes'] = [[bb[0],
                                 w_size - bb[3],
                                 bb[2],
                                 w_size - bb[1]] for bb in sample['bboxes']]

        return sample


class ToTensor(object):
    """Simple wrapper for processing custom input."""
    def __call__(self, sample):
        sample = {'img': F.to_tensor(sample['img']),
                  'bboxes': torch.tensor(sample['bboxes'], dtype=torch.float),
                  'labels': torch.tensor(sample['labels'], dtype=torch.int64)}

        return sample


class Normalize(object):
    """Simple wrapper for processing custom input."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['img'] = F.normalize(
            sample['img'], mean=self.mean, std=self.std
        )
        return sample
