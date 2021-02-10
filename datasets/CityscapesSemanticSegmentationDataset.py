import torch
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from PIL import Image
from PIL import ImageOps
import json
from pycocotools.coco import COCO
import os
import contextlib
import random
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
        self.train = train
        if train:
            self.imgs = glob.glob('data/cityscapes/leftImg8bit/train/*/*.png')  # + \
                        # glob.glob('data/cityscapes/leftImg8bit/train_extra/*/*.png')
            self.labels = glob.glob('data/cityscapes/gtFine/train/*/*gtFine_labelIds.png')  # + \
                          # glob.glob('data/cityscapes/gtCoarse/train_extra/*/*gtCoarse_labelIds.png')
        else:
            self.imgs = glob.glob('data/cityscapes/leftImg8bit/val/*/*.png')
            self.labels = glob.glob('data/cityscapes/gtFine/val/*/*gtFine_labelIds.png')
        self.imgs.sort()
        self.labels.sort(key=lambda x: x.replace("gtCoarse/", "").replace("gtFine", ""))
        if train:
            self.img_preprocessor = transforms.Compose([
                # transforms.Resize((224, 224)),
                # transforms.ColorJitter(.4, .4, .4, .2),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_preprocessor = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.label_preprocessor = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            img = Image.open(self.imgs[idx]).convert('RGB')
            label = Image.open(self.labels[idx]).convert('I')
            ogw, ogh = img.size

            # # Sclae the image
            # scale = random.uniform(0.5, 2.0)
            # img = img.resize((int(ogw * scale), int(ogh * scale)))
            # label = label.resize((int(ogw * scale), int(ogh * scale)), resample=Image.NEAREST)
            #
            # # Mirror the image half of the time
            # if random.uniform(0, 1) > 0.5:
            #     img = ImageOps.mirror(img)
            #     label = ImageOps.mirror(label)

            # Add random crop to image
            cw = 450  # random.randint(200, ogw)
            ch = 450  # min(int(random.uniform(0.5, 1.0) * cw), ogh)  # random.randint(200, ogh)
            x = random.randint(0, ogw - cw)
            y = random.randint(0, ogh - ch)
            img = img.crop((x, y, x+cw, y+ch))
            label = label.crop((x, y, x+cw, y+ch))

            img = self.img_preprocessor(img)
            label = self.label_preprocessor(label).long().squeeze()
        else:
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
