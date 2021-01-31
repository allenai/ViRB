import torch
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from PIL import Image
import json
from pycocotools.coco import COCO
import os
import contextlib
from cityscapesscripts.helpers.labels import labels as cslabels


mapping_20 = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0
    }


class CityscapesSemanticSegmentationDataset:
    """COCO detection dataset class"""

    def __init__(self, train=True):
        super().__init__()
        self.imgs = glob.glob('data/cityscapes/leftImg8bit/%s/*/*.png' % ('train' if train else 'val'))
        self.imgs.sort()
        self.labels = glob.glob('data/cityscapes/gtFine/%s/*/*gtFine_labelIds.png' % ('train' if train else 'val'))
        self.labels.sort()
        self.img_preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(.4, .4, .4, .2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_preprocessor(Image.open(self.imgs[idx]).convert('RGB'))
        label = self.label_preprocessor(Image.open(self.labels[idx]).convert('I')).long().squeeze()
        for cat in torch.unique(label):
            label[label == int(cat)] = mapping_20[int(cat)]
        return img, label

    def __len__(self):
        return len(self.imgs)

    def class_names(self):
        return [l[0] for l in cslabels]

    def num_classes(self):
        return 20
