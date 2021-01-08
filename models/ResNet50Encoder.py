import torch
import torchvision
import torch.nn as nn


class ResNet50Encoder(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        if weights:
            self.model = torchvision.models.resnet50(pretrained=False)
            self.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        res = {}
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        res["layer1"] = x
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        res["layer2"] = x
        x = self.model.layer2(x)
        res["layer3"] = x
        x = self.model.layer3(x)
        res["layer4"] = x
        x = self.model.layer4(x)
        res["layer5"] = x

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        res["embedding"] = x
        return res

    def outputs(self):
        return {
            "embedding": torch.Size([2048]),
            "layer1": torch.Size([2048, 7, 7]),
            "layer2": torch.Size([1024, 56, 56]),
            "layer3": torch.Size([512, 28, 28]),
            "layer4": torch.Size([256, 14, 14]),
            "layer5": torch.Size([64, 7, 7])
        }
