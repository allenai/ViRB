import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
from PIL import Image
import random


class ImagenetEncodableDataset(Dataset):
    """Imagenet encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/imagenet/train/*/*.JPEG' if train else 'data/imagenet/val/*/*.JPEG'
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

        return self.preprocessor(Image.open(self.data[idx])), self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)

