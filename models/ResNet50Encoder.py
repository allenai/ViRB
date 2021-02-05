import torch
import torchvision
import torch.nn as nn


class ResNet50Encoder(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        if weights:
            self.model = torchvision.models.resnet50(pretrained=False)
            self.model.layer4[0].conv2 = nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            self.model.layer4[0].downsample = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            self.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):

        original_batch_size = x.size(0)
        if len(x.shape) == 5:
            x = x.view(-1, x.size(2), x.size(3), x.size(4))

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

        if original_batch_size != x.size(0):
            for name, layer in res.items():
                res[name] = layer.view(original_batch_size, -1, *layer.shape[1:])

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
