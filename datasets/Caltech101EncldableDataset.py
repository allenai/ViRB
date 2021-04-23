import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image
import random
import json
import os
import urllib.request


class CalTech101EncodableDataset(Dataset):
    """Caltech-101 encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        json_file = 'data/caltech-101/caltech101_%s.json' % ('train' if train else 'test')
        if not os.path.exists(json_file):
            url = 'https://prior-datasets.s3.us-east-2.amazonaws.com/ViRB/caltech101_%s.json' % \
                  ('train' if train else 'test')
            urllib.request.urlretrieve(url, json_file)
        with open(json_file) as jf:
            image_list = json.load(jf)
        self.data = [img for img in list(glob.glob('data/caltech-101/*/*.jpg')) if img[17:] in image_list]
        random.shuffle(self.data)
        cats = list(set([path.split("/")[3] for path in self.data]))
        cats.sort()
        self.labels = torch.LongTensor([cats.index(path.split("/")[3]) for path in self.data])
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.preprocessor(Image.open(self.data[idx]).convert('RGB')), self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)
