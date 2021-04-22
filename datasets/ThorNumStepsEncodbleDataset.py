import torch
import tqdm
import torchvision.transforms as transforms
import glob
from PIL import Image


from datasets.EncodableDataset import EncodableDataset


class ThorNumStepsEncodableDataset(EncodableDataset):
    """Pets encodable dataset class"""

    def __init__(self, train=True):
        super().__init__()
        path = 'data/thor_num_steps/train/*/*rgb.jpg' if train else 'data/thor_num_steps/val/*/*rgb.jpg'
        self.data = list(glob.glob(path))
        self.data.sort()
        with open('data/thor_num_steps/%s_labels.txt' % ('train' if train else 'val')) as f:
            lines = f.readlines()
            label_table = {line.split(" ")[0]: int(line.split(" ")[1]) for line in lines}
        self.labels = torch.LongTensor([label_table[img[20:-8]] for img in self.data])
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

    def class_names(self):
        return self.cats

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
