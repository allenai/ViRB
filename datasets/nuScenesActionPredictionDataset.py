import torch
import torchvision.transforms as transforms
import glob
from PIL import Image
import json


class nuScenesActionPredictionDataset:
    """Pets encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        with open('data/nuScenes/nuScenes_%s.json' % ('train' if train else 'test')) as f:
            self.data = json.load(f)
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.actions = list(set([d["label"] for d in self.data]))
        self.actions.sort()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = "data/nuScenes/"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        d = self.data[idx]
        imga = self.preprocessor(Image.open(self.root + d["img_a"]).convert('RGB'))
        imgb = self.preprocessor(Image.open(self.root + d["img_b"]).convert('RGB'))
        return torch.stack((imga, imgb), dim=0), self.actions.index(d["label"])

    def __len__(self):
        return len(self.data)

    def class_names(self):
        return self.actions

    def num_classes(self):
        return len(self.actions)
