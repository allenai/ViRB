import torch
from torch.utils.data import Dataset
import tqdm


class EncodableDataset(Dataset):
    """Encodable dataset class"""

    def __init__(self):
        self.data = []
        self.labels = []
        self.encoded_data = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raise NotImplementedError

    def encode(self, model):
        raise NotImplementedError

    def num_classes(self):
        return int(max(self.labels) + 1)
