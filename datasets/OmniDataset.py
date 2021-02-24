import torch
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from PIL import Image
import random


IMGS_PER_DATASET = 1000
PATHS = {
    'Caltech': 'data/caltech-101/train/*/*.jpg',
    'Cityscapes': 'data/cityscapes/leftImg8bit/train/*/*.png',
    'CLEVR': 'data/CLEVR/images/train/*.png',
    'dtd': 'data/dtd/train/*/*.jpg',
    'Egohands': 'data/egohands/images/*',
    'Eurosat': 'data/eurosat/train/*/*.jpg',
    'ImageNet': 'data/imagenet/train/*/*.JPEG',
    'Kinetics': 'data/kinetics400/*/*.jpg',
    'KITTI': 'data/KITTI/training/image_2/*.png',
    'nuScenes': 'data/nuScenes/samples/CAM_FRONT/*.jpg',
    'NYU': 'data/nyu/train/images/*.png',
    'Pets': 'data/pets/train/*/*.jpg',
    'SUN397': 'data/SUN397/train/*/*.jpg',
    'Taskonomy': 'data/taskonomy/train/rgb/*/*.png',
    'ThorActionPrediction': 'data/thor_action_prediction/train/*/*.jpg'
}


class OmniDataset:
    """Class of every dataset"""

    def __init__(self, keys):
        super().__init__()
        self.data = []
        for key in keys:
            path = PATHS[key]
            imgs = glob.glob(path)
            imgs.sort()
            random.seed(1999)
            random.shuffle(imgs)
            for i in range(IMGS_PER_DATASET):
                self.data.append(imgs[i])
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.preprocessor(Image.open(self.data[idx]).convert('RGB'))

    def __len__(self):
        return len(self.data)
