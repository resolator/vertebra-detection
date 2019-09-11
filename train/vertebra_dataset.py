#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json

import torchvision
from torch.utils.data import Dataset

sys.path.append(os.path.join(sys.path[0], '../'))
from common.transforms import ToTensor


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
