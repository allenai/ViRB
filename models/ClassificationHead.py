import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.head(x)
