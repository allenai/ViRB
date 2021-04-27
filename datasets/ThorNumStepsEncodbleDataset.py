import torch
from torch.utils.data import Dataset
import tqdm
import torchvision.transforms as transforms
import glob
from PIL import Image


class ThorNumStepsEncodableDataset(Dataset):
    """Thor NumSteps encodable dataset class"""

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

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)
