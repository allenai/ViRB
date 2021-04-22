import torch
import tqdm
import torchvision.transforms as transforms
import glob
from PIL import Image
import random


from datasets.EncodableDataset import EncodableDataset


class ImagenetEncodableDataset(EncodableDataset):
    """Imagenet encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/imagenet/train/*/*.JPEG' if train else 'data/imagenet/val/*/*.JPEG'
        self.data = list(glob.glob(path))
        random.shuffle(self.data)
        cats = list(set([path.split("/")[3] for path in self.data]))
        cats.sort()
        self.labels = torch.LongTensor([cats.index(path.split("/")[3]) for path in self.data])
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
