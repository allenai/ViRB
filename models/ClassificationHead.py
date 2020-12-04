import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.embedding_size = embedding_size
        # self.head = nn.Sequential(
        #     nn.Linear(embedding_size, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, output_size)
        # )
        self.head = nn.Linear(embedding_size, output_size)

    def forward(self, x):
        return self.head(x["embedding"].float())

    def required_encoding(self):
        return ["embedding"]
