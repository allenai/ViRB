import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
from PIL import Image
import random


class SUN397EncodableDataset(Dataset):
    """SUN397 encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/SUN397/train/*/*.jpg' if train else 'data/SUN397/test/*/*.jpg'
        self.data = list(glob.glob(path))
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

        if len(self.encoded_data) == 0:
            return self.preprocessor(Image.open(self.data[idx]).convert('RGB')), self.labels[idx]
        return self.encoded_data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)
