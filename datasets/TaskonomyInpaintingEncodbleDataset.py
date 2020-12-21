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


class TaskonomyInpaintingEncodableDataset(EncodableDataset):
    """Taskonomy inpainting encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        data_path = 'data/taskonomy/train/rgb/*/*.png'\
            if train else 'data/taskonomy/test/rgb/*/*.png'
        self.data = list(glob.glob(data_path))
        self.data.sort()
        self.labels = self.data
        self.img_preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.data[idx]

        img = self.img_preprocessor(Image.open(img_path).convert('RGB'))
        mask = img.detach().clone()
        img[:,64:160, 64:160] = 0.0

        # i = img.detach().numpy().transpose(1, 2, 0)
        # plt.figure(0)
        # plt.imshow(i)
        # m = mask.detach().numpy().transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(m)
        # plt.show()
        # exit()

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
