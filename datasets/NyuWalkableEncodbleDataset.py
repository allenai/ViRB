import torch
import tqdm
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image


from datasets.EncodableDataset import EncodableDataset


class NyuWalkableEncodableDataset(EncodableDataset):
    """NYU Depth encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        self.train = train
        data_path = 'data/nyu/train/images/*.png'\
            if train else 'data/nyu/test/images/*.png'
        self.data = list(glob.glob(data_path))
        self.data.sort()
        label_path = 'data/nyu/train/walkable/*.png'\
            if train else 'data/nyu/test/walkable/*.png'
        self.labels = list(glob.glob(label_path))
        self.labels.sort()
        if train:
            self.img_preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(.4, .4, .4, .2),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.label_preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.25)
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.data[idx]
        label_path = self.labels[idx]

        # img = Image.open(img_path).convert('RGB')
        # mask = Image.open(label_path)
        # if self.train:
        #     # Add random crop to image
        #     ogw, ogh = img.size
        #     cw = 412
        #     ch = 412
        #     x = random.randint(0, ogw - cw)
        #     y = random.randint(0, ogh - ch)
        #     img = img.crop((x, y, x+cw, y+ch))
        #     mask = mask.crop((x, y, x+cw, y+ch))
        # img = self.img_preprocessor(img)
        # mask = np.array(mask, dtype=np.float)
        # mask[mask != 0.0] = 1.0
        # mask = torch.tensor(mask, dtype=torch.float)
        # mask = mask.unsqueeze(0)

        img = self.img_preprocessor(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(label_path).resize((224, 224)), dtype=np.float)
        mask[mask != 0.0] = 1.0

        mask = torch.tensor(mask, dtype=torch.float)
        mask = mask.unsqueeze(0)

        return img, mask

    def __len__(self):
        return len(self.data)

    def num_classes(self):
        return 1

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
