import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image


class ThorDepthEncodableDataset(Dataset):
    """Thor depth encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        data_path = 'data/thor_depth_prediction/train/*/*rgb.jpg'\
            if train else 'data/thor_depth_prediction/test/*/*rgb.jpg'
        self.data = list(glob.glob(data_path))
        self.data.sort()
        label_path = 'data/thor_depth_prediction/train/*/*depth.jpg'\
            if train else 'data/thor_depth_prediction/test/*/*depth.jpg'
        self.labels = list(glob.glob(label_path))
        self.labels.sort()
        if train:
            self.img_preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(.4, .4, .4, .2),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.label_preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.25)
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.data[idx]
        label_path = self.labels[idx]

        img = self.img_preprocessor(Image.open(img_path).convert('RGB'))
        mask = self.label_preprocessor(Image.open(label_path).convert('L'))

        return img, mask

    def __len__(self):
        return len(self.data)

    def num_classes(self):
        return int(max(self.labels) + 1)
