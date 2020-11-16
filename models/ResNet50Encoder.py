import torch
import torchvision
import torch.nn as nn


class ResNet50Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.encoder(x)
