import torch
import torchvision.transforms as transforms
import glob
from PIL import Image
import random


PATHS = {
    'Caltech': 'data/omni_10k/Caltech/*.png',
    'Cityscapes': 'data/omni_10k/Cityscapes/*.png',
    'CLEVR': 'data/omni_10k/CLEVR/*.png',
    'dtd': 'data/omni_10k/dtd/*.png',
    'Egohands': 'data/omni_10k/Egohands/*.png',
    'Eurosat': 'data/omni_10k/Eurosat/*.png',
    'ImageNet': 'data/omni_10k/ImageNet/*.png',
    'Kinetics': 'data/omni_10k/Kinetics/*.png',
    'KITTI': 'data/omni_10k/KITTI/*.png',
    'nuScenes': 'data/omni_10k/nuScenes/*.png',
    'NYU': 'data/omni_10k/NYU/*.png',
    'Pets': 'data/omni_10k/Pets/*.png',
    'SUN397': 'data/omni_10k/SUN397/*.png',
    'Taskonomy': 'data/omni_10k/Taskonomy/*.png',
    'Thor': 'data/omni_10k/Thor/*.png'
}


class OmniDataset:
    """Class of every dataset"""

    def __init__(self, keys, max_imgs=10000, resize=(224, 224)):
        super().__init__()
        self.data = []
        keys = keys if isinstance(keys, list) else [keys]
        for key in keys:
            path = PATHS[key]
            imgs = glob.glob(path)
            imgs.sort()
            random.seed(1999)
            random.shuffle(imgs)
            print(key, len(imgs))
            for i in range(min(len(imgs), max_imgs)):
                self.data.append(imgs[i])
        self.preprocessor = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.preprocessor(Image.open(self.data[idx]).convert('RGB'))

    def __len__(self):
        return len(self.data)
