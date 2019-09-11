#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torchvision.transforms import functional as F

import albumentations as alb
import albumentations.pytorch as alb_pytorch
from albumentations import BasicTransform


class ToTensor(BasicTransform):
    """Simple wrapper for processing custom input."""
    def __init__(self, normalize=None):
        super(ToTensor, self).__init__(always_apply=True, p=1)
        self.alb_totensor = alb_pytorch.ToTensor()
        self.normalize = normalize

    def __call__(self, force_apply=False, **kwargs):
        sample = {'image': self.alb_totensor(**kwargs)['image'],
                  'bboxes': torch.tensor(kwargs['bboxes'], dtype=torch.float),
                  'labels': torch.tensor(kwargs['labels'], dtype=torch.int64)}
        if self.normalize is not None:
            sample['image'] = F.normalize(sample['image'], **self.normalize)

        return sample

    def get_transform_init_args_names(self):
        return ['normalize']


def get_test_transform(resize_size,
                       center_crop_size,
                       min_visibility=0.0):
    """Create transform for test phase.

    Parameters
    ----------
    resize_size : array-like
        Target size (height, width).
    center_crop_size : array-like
        Target size for center crop (height, width).
    min_visibility : float
        Min fraction of area for a bounding box to remain this box in list.

    Returns
    -------
    transforms : albumentation.Compose
        Composed test transform.
    bbox_params : albumentation.BboxParams
        Bounding box parameters for create your own transform.

    """
    bbox_params = alb.BboxParams(
        format='pascal_voc',
        min_visibility=min_visibility,
        label_fields=['labels']
    )

    transform = alb.Compose([
        alb.Resize(resize_size[0], resize_size[1]),
        alb.CenterCrop(center_crop_size[0], center_crop_size[1])
    ], bbox_params=bbox_params)

    return transform, bbox_params
