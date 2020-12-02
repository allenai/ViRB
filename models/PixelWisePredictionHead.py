import torch.nn as nn
import torch.nn.functional as F


class PixelWisePredictionHead(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.up1 = upshuffle(2048, 1024, 2, kernel_size=3, stride=1, padding=1)
        self.up2 = upshuffle(1024, 512, 2, kernel_size=3, stride=1, padding=1)
        self.up3 = upshuffle(512, 256, 2, kernel_size=3, stride=1, padding=1)
        self.up4 = upshuffle(256, 64, 2, kernel_size=3, stride=1, padding=1)
        self.up5 = upshufflenorelu(64, output_size, 2)

    def forward(self, x):
        d5 = self.up1(x["layer5"])
        d5_ = _upsample_add(d5, x["layer4"])
        d4 = self.up2(d5_)
        d4_ = _upsample_add(d4, x["layer3"])
        d3 = self.up3(d4_)
        d3_ = _upsample_add(d3, x["layer3"])
        d2 = self.up4(d3_)
        d2_ = _upsample_add(d2, x["layer1"])
        out = self.up5(d2_)
        print("OUT SHAPE:", out.shape)
        exit()
        out = out.permute(0, 2, 3, 1).flatten()
        return out

    def required_encoding(self):
        return ["layer1", "layer2", "layer3", "layer4", "layer5"]


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y


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
