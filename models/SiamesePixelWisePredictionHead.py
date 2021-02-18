import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamesePixelWisePredictionHead(nn.Module):

    def __init__(self, output_size):
        super().__init__()

        self.up1 = upshuffle(2048, 1024, 2, kernel_size=3, stride=1, padding=1)
        self.up2 = upshuffle(1024, 512, 2, kernel_size=3, stride=1, padding=1)
        self.up3 = upshuffle(512, 256, 2, kernel_size=3, stride=1, padding=1)
        self.up4 = upshuffle(256, 64, 2, kernel_size=3, stride=1, padding=1)
        self.up5 = upshufflenorelu(64, output_size, 2)

        self.fusion1 = nn.Conv2d(4096, 2048, 1)
        self.fusion2 = nn.Conv2d(2048, 1024, 1)
        self.fusion3 = nn.Conv2d(1024, 512, 1)
        self.fusion4 = nn.Conv2d(512, 256, 1)
        self.fusion5 = nn.Conv2d(128, 64, 1)

    def forward(self, x):
        x["layer5"] = self.fusion1(x["layer5"].view(x["layer5"].size(0), -1, x["layer5"].size(3), x["layer5"].size(4)))
        x["layer4"] = self.fusion2(x["layer4"].view(x["layer4"].size(0), -1, x["layer4"].size(3), x["layer4"].size(4)))
        x["layer3"] = self.fusion3(x["layer3"].view(x["layer3"].size(0), -1, x["layer3"].size(3), x["layer3"].size(4)))
        x["layer2"] = self.fusion4(x["layer2"].view(x["layer2"].size(0), -1, x["layer2"].size(3), x["layer2"].size(4)))
        x["layer1"] = self.fusion5(x["layer1"].view(x["layer1"].size(0), -1, x["layer1"].size(3), x["layer1"].size(4)))
        d5 = self.up1(x["layer5"])
        d5_ = _upsample_add(d5, x["layer4"])
        d4 = self.up2(d5_)
        d4_ = _upsample_add(d4, x["layer3"])
        d3 = self.up3(d4_)
        d3_ = _upsample_add(d3, x["layer2"])
        d2 = self.up4(d3_)
        d2_ = _upsample_add(d2, x["layer1"])
        out = self.up5(d2_)
        return out

    def required_encoding(self):
        return ["layer1", "layer2", "layer3", "layer4", "layer5"]


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y


def _upsample(x, factor):
    _, _, H, W = x.size()
    return F.upsample(x, size=(H*factor, W*factor), mode='bilinear')


def upshuffle(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
        nn.LeakyReLU()
    )


def upshufflenorelu(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
    )
