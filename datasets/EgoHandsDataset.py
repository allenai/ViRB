import torch
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageOps, ImageDraw
import json
from pycocotools.coco import COCO
import os
import contextlib
import random
from scipy.io import loadmat
import numpy as np


## Splits from 2015 EgoHands Paper
SPLITS = {
    "TEST":  [1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
    "VALID": [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    "TRAIN": [0,1,1,1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1]
}


class EgoHandsDataset:
    """COCO detection dataset class"""

    def __init__(self, train=True):
        super().__init__()
        self.train = train
        img_dirs = glob.glob('data/egohands/images/*')
        img_dirs.sort()
        label_dirs = glob.glob('data/egohands/labels/*')
        label_dirs.sort()

        if self.train:
            pruned_img_dirs = [img for img, si in zip(img_dirs, SPLITS["TRAIN"]) if si == 1]
            self.imgs = [img for pid in pruned_img_dirs for img in glob.glob(pid+"/*.jpg")]
            pruned_label_dirs = [label for label, si in zip(label_dirs, SPLITS["TRAIN"]) if si == 1]
            self.labels = [label for pld in pruned_label_dirs for label in glob.glob(pld + "/*.png")]
        else:
            pruned_img_dirs = [img for img, si in zip(img_dirs, SPLITS["TEST"]) if si == 1]
            self.imgs = [img for pid in pruned_img_dirs for img in glob.glob(pid + "/*.jpg")]
            pruned_label_dirs = [label for label, si in zip(label_dirs, SPLITS["TEST"]) if si == 1]
            self.labels = [label for pld in pruned_label_dirs for label in glob.glob(pld + "/*.png")]
        self.imgs.sort()
        self.labels.sort()

        self.img_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_preprocessor = transforms.Compose([
            transforms.ToTensor()
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            i = Image.open(self.imgs[idx]).convert('RGB')
            l = Image.open(self.labels[idx]).convert('I')
            ogw, ogh = i.size

            # # Scale the image
            # scale = random.uniform(0.5, 2.0)
            # img = img.resize((int(ogw * scale), int(ogh * scale)))
            # label = label.resize((int(ogw * scale), int(ogh * scale)), resample=Image.NEAREST)
            #
            # # Mirror the image half of the time
            # if random.uniform(0, 1) > 0.5:
            #     img = ImageOps.mirror(img)
            #     label = ImageOps.mirror(label)

            # Add random crop to image

            repeat_counter = 0
            while True:
                cw = 513
                ch = 513
                x = random.randint(0, ogw - cw)
                y = random.randint(0, ogh - ch)
                img = i.crop((x, y, x+cw, y+ch))
                label = l.crop((x, y, x+cw, y+ch))

                img = self.img_preprocessor(img)
                label = self.label_preprocessor(label).long().squeeze()

                repeat_counter += 1
                if len(torch.unique(label)) > 1 or repeat_counter > 10:
                    break

        else:
            img = self.img_preprocessor(Image.open(self.imgs[idx]).convert('RGB'))
            label = self.label_preprocessor(Image.open(self.labels[idx]).convert('I')).long().squeeze()

        return img, label

    def __len__(self):
        return len(self.imgs)

    def class_names(self):
        return ["background", "yourleft", "yourright", "myleft", "myright"]

    def num_classes(self):
        return 5
