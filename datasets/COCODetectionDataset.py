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


from datasets.EncodableDataset import EncodableDataset


class COCODetectionDataset:
    """COCO detection dataset class"""

    def __init__(self, train=True):
        super().__init__()
        self.imgs = glob.glob('data/coco/%s2017/*.jpg' % ('train' if train else 'val'))
        self.imgs.sort()
        self.labels = glob.glob('data/coco/annotations/panoptic_labels_%s2017/*.png' % ('train' if train else 'val'))
        self.labels.sort()
        self.img_preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
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

        img = Image.open(self.imgs[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('I')
        return self.img_preprocessor(img), self.label_preprocessor(label).long().squeeze()

    def __len__(self):
        return len(self.imgs)

    def class_names(self):
        return ["apple"]

    def num_classes(self):
        return 200
