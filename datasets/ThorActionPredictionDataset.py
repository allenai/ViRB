import torch
import torchvision.transforms as transforms
import glob
from PIL import Image


ACTIONS = {'MoveAhead': 0, 'RotateLeft': 1, 'RotateRight': 2, 'LookUp': 3, 'LookDown': 4}


class ThorActionPredictionDataset:
    """Pets encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        with open('data/thor_action_prediction/%s_labels.txt' % ('train' if train else 'val')) as f:
            lines = f.readlines()
            self.data = [tuple(line.replace("\n", "").split(" ")) for line in lines]
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, action = self.data[idx]
        img_name = "data/thor_action_prediction/" + img_name
        imga = self.preprocessor(Image.open(img_name+"_rgb_a.jpg").convert('RGB'))
        imgb = self.preprocessor(Image.open(img_name+"_rgb_b.jpg").convert('RGB'))
        return torch.stack((imga, imgb), dim=0), torch.LongTensor([ACTIONS[action]])

    def __len__(self):
        return len(self.data)

    def class_names(self):
        return list(ACTIONS.keys())

    def num_classes(self):
        return len(ACTIONS)
