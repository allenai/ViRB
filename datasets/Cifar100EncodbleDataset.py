import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFAR100EncodableDataset(Dataset):
    """CIFAR-100 encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/cifar-100/train' if train else 'data/cifar-100/test'
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        self.data = dict[list(dict.keys())[4]]
        self.labels = dict[list(dict.keys())[2]]
        self.preprocessor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data[idx].reshape((3, 32, 32))
        x = np.moveaxis(x, 0, 2)
        x = self.preprocessor(x)
        return x, self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)
