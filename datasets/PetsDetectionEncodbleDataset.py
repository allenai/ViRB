import torch
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from PIL import Image
import random


from datasets.EncodableDataset import EncodableDataset


class PetsDetectionEncodableDataset(EncodableDataset):
    """Pets encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/pets/train/*/*.jpg' if train else 'data/pets/test/*/*.jpg'
        self.data = list(glob.glob(path))
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
        mask = torch.FloatTensor(np.array(Image.open(mask_path).resize((224, 224)))).unsqueeze(0).half()
        mask -= 1
        mask[mask == 2] = 1

        img = self.preprocessor(Image.open(path).convert('RGB')).half()

        return img, mask

    def __len__(self):
        return len(self.data)

    def encode(self, model):
        model.to(self.device)
        model.eval()
        batch = []
        for img in tqdm.tqdm(self.data):
            if len(batch) == 500:
                batch = torch.stack(batch, dim=0).to(self.device)
                with torch.no_grad():
                    out = model(batch).detach()
                self.encoded_data.append(out)
                batch = []
            x = Image.open(img).convert('RGB')
            x = self.preprocessor(x)
            batch.append(x)
        batch = torch.stack(batch, dim=0).to(self.device)
        with torch.no_grad():
            out = model(batch).detach()
        self.encoded_data.append(out)
        self.encoded_data = torch.cat(self.encoded_data, dim=0).squeeze().to("cpu")
