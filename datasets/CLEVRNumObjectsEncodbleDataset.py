import torch
import torchvision.transforms as transforms
import glob
from PIL import Image
import json
from torch.utils.data import Dataset


class CLEVRNumObjectsEncodableDataset(Dataset):
    """CLEVR Num Objects encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/CLEVR/images/train/*.png' if train else 'data/CLEVR/images/val/*.png'
        self.data = glob.glob(path)
        self.data.sort()
        labels_path = 'data/CLEVR/scenes/CLEVR_train_scenes.json' if train else \
            'data/CLEVR/scenes/CLEVR_val_scenes.json'
        with open(labels_path) as f:
            scene_data = json.load(f)
        self.labels = torch.LongTensor([len(s['objects']) for s in scene_data['scenes']])
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
