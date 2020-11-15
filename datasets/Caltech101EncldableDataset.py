import torch
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from PIL import Image

from datasets.EncodableDataset import EncodableDataset


class CalTech101EncodableDataset(EncodableDataset):
    """CIFAR-100 encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/caltech-101/train/*/*.jpg' if train else 'data/caltech-101/test/*/*.jpg'
        self.data = list(glob.glob(path))
        cats = list(set([path.split("/")[3] for path in self.data]))
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

        if len(self.encoded_data) == 0:
            return self.data[idx], self.labels[idx]
        return self.encoded_data[idx], self.labels[idx]

    def encode(self, model):
        model.to(self.device)
        model.eval()
        batch = []
        for img in tqdm.tqdm(self.data):
            if len(batch) < 100:
                x = Image.open(img).convert('RGB')
                x = self.preprocessor(x)
                batch.append(x)
            else:
                batch = torch.stack(batch, dim=0).to(self.device)
                out = model(batch).detach()
                self.encoded_data.append(out)
                batch = []
        batch = torch.stack(batch, dim=0).to(self.device)
        out = model(batch).detach()
        self.encoded_data.append(out)
        self.encoded_data = torch.cat(self.encoded_data, dim=0).squeeze().to("cpu")
        print("Encoded data shape:", self.encoded_data.shape, "label shape:", self.labels.shape)