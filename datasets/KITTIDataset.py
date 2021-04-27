import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
from PIL import Image
import random


class KITTIDataset(Dataset):
    """KITTI optical flow dataset"""

    def __init__(self, train=True):
        super().__init__()
        self.train = train
        self.imgs_a = glob.glob('data/KITTI/training/image_2/*_10.png')
        self.imgs_a.sort()
        self.imgs_b = glob.glob('data/KITTI/training/image_2/*_11.png')
        self.imgs_b.sort()
        self.labels = glob.glob('data/KITTI/training/viz_flow_occ/*_10.png')
        self.labels.sort()
        if train:
            self.imgs_a = self.imgs_a[:int(0.9*len(self.imgs_a))]
            self.imgs_b = self.imgs_b[:int(0.9 * len(self.imgs_b))]
            self.labels = self.labels[:int(0.9 * len(self.labels))]
        else:
            self.imgs_a = self.imgs_a[int(0.9 * len(self.imgs_a)):]
            self.imgs_b = self.imgs_b[int(0.9 * len(self.imgs_b)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]
        if train:
            self.img_preprocessor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_preprocessor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.label_preprocessor = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            img_a = Image.open(self.imgs_a[idx]).convert('RGB')
            img_b = Image.open(self.imgs_b[idx]).convert('RGB')
            label = Image.open(self.labels[idx]).convert('F').resize(img_a.size)
            ogw, ogh = img_a.size

            # Add crop to image
            cw = 224
            ch = 224
            x = random.randint(0, ogw - cw)
            y = random.randint(0, ogh - ch)
            img_a = img_a.crop((x, y, x+cw, y+ch))
            img_b = img_b.crop((x, y, x + cw, y + ch))
            label = label.crop((x, y, x+cw, y+ch))

            img_a = self.img_preprocessor(img_a)
            img_b = self.img_preprocessor(img_b)
            label = self.label_preprocessor(label).squeeze()
        else:
            img_a = self.img_preprocessor(Image.open(self.imgs_a[idx]).convert('RGB'))
            img_b = self.img_preprocessor(Image.open(self.imgs_b[idx]).convert('RGB'))
            img_size = 1242, 376
            label = self.label_preprocessor(Image.open(self.labels[idx]).convert('F').resize(img_size)).squeeze()
        return torch.stack((img_a, img_b), dim=0), label / 255

    def __len__(self):
        return len(self.imgs_a)

