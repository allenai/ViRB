import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
from PIL import Image


class PetsDetectionEncodableDataset(Dataset):
    """Pets Detection encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/pets/train/*/*.jpg' if train else 'data/pets/test/*/*.jpg'
        self.data = list(glob.glob(path))
        if train:
            self.preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(.4, .4, .4, .2),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.data[idx]

        mask_path = "data/pets/annotations/trimaps/" + path.split("/")[-1].replace("jpg", "png")
        mask = torch.FloatTensor(np.array(Image.open(mask_path).resize((224, 224)))).unsqueeze(0)
        mask -= 1
        mask[mask == 2] = 1

        img = self.preprocessor(Image.open(path).convert('RGB'))

        return img, mask

    def __len__(self):
        return len(self.data)

    def num_classes(self):
        return 1
