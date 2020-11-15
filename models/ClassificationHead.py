import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.head = nn.Linear(embedding_size, output_size)

    def forward(self, x):
        return self.head(x)
