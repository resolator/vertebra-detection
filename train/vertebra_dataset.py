#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import json

import albumentations.pytorch as alb_pytorch
from albumentations import BasicTransform

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


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


class VertebraDataset(Dataset):
    """Custom dataset for FasterRCNN.

    Attributes
    ----------
    transform : torchvision.transforms
        Custom transforms.
    samples : List
        Loaded samples information from json file.

    """
    def __init__(self, json_path, transform=None):
        with open(json_path) as f:
            self.samples = json.load(f)

        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labels = [int(x['label']) for x in self.samples[idx]['annotation']]
        bboxes = [x['bbox'] for x in self.samples[idx]['annotation']]
        img = cv2.imread(self.samples[idx]['img_path'])
        sample = {'image': img, 'bboxes': bboxes, 'labels': labels}
        sample = self.transform(**sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        return batch
