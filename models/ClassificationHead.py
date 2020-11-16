import torch.nn as nn
import torch
import torchvision


class ClassificationHead(nn.Module):

    def __init__(self, embedding_size, output_size):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        return self.head(x)
