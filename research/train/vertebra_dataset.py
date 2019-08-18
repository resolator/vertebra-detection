#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class Resize(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def bboxes_resize(self, img_size, bboxes):
        w_factor = self.width / img_size[0]
        h_factor = self.height / img_size[1]

        return [[int(bb[0] * w_factor),
                 int(bb[1] * h_factor),
                 int(bb[2] * w_factor),
                 int(bb[3] * h_factor)] for bb in bboxes]

    def __call__(self, sample):
        sample['bboxes'] = self.bboxes_resize(
            sample['img'].size, sample['bboxes']
        )
        sample['img'] = F.resize(sample['img'], (self.height, self.width))

        return sample


class BottomRightCrop(object):
    def __init__(self, crop_factor):
        self.crop_factor = crop_factor

    def __call__(self, sample):
        x = self.crop_factor * sample['img'].size[0]
        y = self.crop_factor * sample['img'].size[1]
        w = sample['img'].size[0] - x
        h = sample['img'].size[1] - y

        sample['img'] = F.crop(sample['img'], x, y, h, w)
        sample['bboxes'] = [[round(bb[0] - x),
                             round(bb[1] - y),
                             round(bb[2] - x),
                             round(bb[3] - y)] for bb in sample['bboxes']]

        return sample


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['img'] = F.hflip(sample['img'])
            h_size = sample['img'].size[1]
            sample['bboxes'] = [[h_size - bb[0],
                                 bb[1],
                                 h_size - bb[2],
                                 bb[3]] for bb in sample['bboxes']]

        return sample


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['img'] = F.vflip(sample['img'])
            w_size = sample['img'].size[0]
            sample['bboxes'] = [[bb[0],
                                 w_size - bb[1],
                                 bb[2],
                                 w_size - bb[3]] for bb in sample['bboxes']]

        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample = {'img': F.to_tensor(sample['img']),
                  'bboxes': torch.tensor(sample['bboxes'], dtype=torch.float),
                  'labels': torch.tensor(sample['labels'], dtype=torch.int64)}

        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['img'] = F.normalize(
            sample['img'], mean=self.mean, std=self.std
        )
        return sample


class VertebraDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path) as f:
            self.samples = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labels = [int(x['label']) for x in self.samples[idx]['annotation']]
        bboxes = [x['bbox'] for x in self.samples[idx]['annotation']]
        img = Image.open(self.samples[idx]['img_path'])
        sample = {'img': img, 'bboxes': bboxes, 'labels': labels}

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = ToTensor()(sample)

        return sample['img'], {'boxes': sample['bboxes'],
                               'labels': sample['labels']}
