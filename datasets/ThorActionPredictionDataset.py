import torch
import torchvision.transforms as transforms
import glob
from PIL import Image


ACTIONS = {'MoveAhead': 0, 'RotateLeft': 1, 'RotateRight': 2, 'LookUp': 3, 'LookDown': 4}


class ThorActionPredictionDataset:
    """Pets encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/thor_action_prediction/%s/*/*rgb_a.jpg' % ('train' if train else 'val')
        self.data = list([p[:-6] for p in glob.glob(path)])
        self.data.sort()
        with open('data/thor_action_prediction/%s_labels.txt' % ('train' if train else 'val')) as f:
            lines = f.readlines()
            self.labels = torch.LongTensor([ACTIONS[line.strip()] for line in lines])
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imga = self.preprocessor(Image.open(self.data[idx]+"_a.jpg").convert('RGB'))
        imgb = self.preprocessor(Image.open(self.data[idx]+"_b.jpg").convert('RGB'))
        return torch.stack((imga, imgb), dim=0), self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def class_names(self):
        return list(ACTIONS.keys())

    def num_classes(self):
        return len(ACTIONS)
