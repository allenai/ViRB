import torch
import torchvision.transforms as transforms
import glob
import json
from PIL import Image


class KineticsActionPredictionDataset:
    """KineticsActionPrediction encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        with open('data/kinetics400/annotations/%s.json' % ('train' if train else 'validate')) as f:
            json_data = json.load(f)
        self.cats = list(set(d["annotations"]["label"] for d in json_data.values()))
        self.cats.sort()
        self.root = 'data/kinetics400/%s' % 'train' if train else 'validate'
        self.data = []
        for img, data in json_data.items():
            imgs = glob.glob("%s/%s/%s_*" % (self.root, img[:2], img))
            if len(imgs) >= 6:
                imgs = imgs[:6]
                imgs.sort()
                self.data.append((imgs, self.cats.index(data["annotations"]["label"])))
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_names, action = self.data[idx]
        imgs = []
        for img_name in img_names:
            imgs.append(self.preprocessor(Image.open(img_name).convert('RGB')))
        return torch.stack(imgs, dim=0), action

    def __len__(self):
        return len(self.data)

    def class_names(self):
        return self.cats

    def num_classes(self):
        return len(self.cats)
