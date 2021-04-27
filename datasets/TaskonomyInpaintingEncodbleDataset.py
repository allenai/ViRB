import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image


class TaskonomyInpaintingEncodableDataset(Dataset):
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

        try:
            img = self.img_preprocessor(Image.open(img_path).convert('RGB'))
        except:
            print(img_path)
            import time
            time.sleep(10)
            print(img_path)
            exit()
        mask = img.detach().clone()
        img[:, 64:160, 64:160] = 0.0

        return img, mask

    def __len__(self):
        return len(self.data)

    def num_classes(self):
        return int(max(self.labels) + 1)
