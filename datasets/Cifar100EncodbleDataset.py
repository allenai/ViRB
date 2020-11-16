import torch
import tqdm
import pickle
import numpy as np
import torchvision.transforms as transforms

from datasets.EncodableDataset import EncodableDataset


class CIFAR100EncodableDataset(EncodableDataset):
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

        if len(self.encoded_data) == 0:
            x = self.data[idx].reshape((3, 32, 32))
            x = np.moveaxis(x, 0, 2)
            x = self.preprocessor(x)
            return x, self.labels[idx]
        return self.encoded_data[idx], self.labels[idx]

    def encode(self, model):
        model.to(self.device)
        for i in tqdm.tqdm(range(len(self.data))):
            x = self.data[i].reshape((3, 32, 32))
            x = np.moveaxis(x, 0, 2)
            x = self.preprocessor(x)
            x = model(x.unsqueeze(0))
            self.encoded_data.append(x)
        self.encoded_data = torch.stack(self.encoded_data, 0)
