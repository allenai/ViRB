import torch
import torch.nn as nn
import torch.nn.functional as F


# class PixelWisePredictionHead(nn.Module):
#
#     def __init__(self, output_size):
#         super().__init__()
#         self.up1 = upshuffle(2048, 1024, 2, kernel_size=3, stride=1, padding=1)
#         self.up2 = upshuffle(1024, 512, 2, kernel_size=3, stride=1, padding=1)
#         self.up3 = upshuffle(512, 256, 2, kernel_size=3, stride=1, padding=1)
#         self.up4 = upshuffle(256, 64, 2, kernel_size=3, stride=1, padding=1)
#         self.up5 = upshufflenorelu(64, output_size, 2)
#
#     def forward(self, x):
#         d5 = self.up1(x["layer5"].float())
#         d5_ = _upsample_add(d5, x["layer4"].float())
#         d4 = self.up2(d5_)
#         d4_ = _upsample_add(d4, x["layer3"].float())
#         d3 = self.up3(d4_)
#         d3_ = _upsample_add(d3, x["layer2"].float())
#         d2 = self.up4(d3_)
#         d2_ = _upsample_add(d2, x["layer1"].float())
#         out = self.up5(d2_)
#         return out
#
#     def required_encoding(self):
#         return ["layer1", "layer2", "layer3", "layer4", "layer5"]
#
#
# def _upsample_add(x, y):
#     _, _, H, W = y.size()
#     return F.upsample(x, size=(H, W), mode='bilinear') + y
#
#
# def _upsample(x, factor):
#     _, _, H, W = x.size()
#     return F.upsample(x, size=(H*factor, W*factor), mode='bilinear')
#
#
# def upshuffle(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
#         nn.PixelShuffle(upscale_factor),
#         nn.LeakyReLU()
#     )
#
#
# def upshufflenorelu(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
#         nn.PixelShuffle(upscale_factor),
#     )


class DeepLabHead(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.aspp = ASPP(2048, 256, num_classes)
        self.low_level_feature_reducer = nn.Sequential(
            nn.Conv2d(256, 16, 1),
            nn.BatchNorm2d(16, momentum=0.0003),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 16, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=0.0003),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256, momentum=0.0003),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256, momentum=0.0003),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_aspp = self.aspp(x["layer5"])
        x_aspp = nn.Upsample((56, 56), mode='bilinear', align_corners=True)(x_aspp)
        x = torch.cat((self.low_level_feature_reducer(x["layer2"]), x_aspp), dim=1)
        x = self.decoder(x)
        x = nn.Upsample((224, 224), mode='bilinear', align_corners=True)(x)
        return x


class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.conv3(x)

        return x

