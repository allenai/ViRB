import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ResNet50Encoder import Bottleneck


class DeepLabHead(nn.Module):

    def __init__(self, num_classes, encoder):
        super().__init__()
        self.aspp = ASPP(2048, 256)
        self.low_level_feature_reducer = nn.Sequential(
            nn.Conv2d(256, 48, 1),
            nn.BatchNorm2d(48, momentum=0.0003),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=0.0003),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=0.0003),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 3, padding=1),
        )
        block4_weights = encoder.model.layer4.state_dict()
        renamed_weights = {}
        for name, weight in block4_weights.items():
            if name[:2] == "0.":
                name = "upsample_layer." + name[2:]
            elif name[:2] == "1.":
                name = "conv.0." + name[2:]
            elif name[:2] == "2.":
                name = "conv.1." + name[2:]
            renamed_weights[name] = weight

        self.block4 = CascadeBlock(Bottleneck, 512, 1024, 3, stride=1, dilation=2)
        self.block4.load_state_dict(renamed_weights)
        # self.block5 = CascadeBlock(Bottleneck, 512, 1024, 3, stride=1, dilation=4)
        # self.block5.load_state_dict(renamed_weights)
        # self.block6 = CascadeBlock(Bottleneck, 512, 1024, 3, stride=1, dilation=8)
        # self.block6.load_state_dict(renamed_weights)
        # self.block7 = CascadeBlock(Bottleneck, 512, 1024, 3, stride=1, dilation=16)
        # self.block7.load_state_dict(renamed_weights)

    def forward(self, x):
        l2_size = tuple(x["layer2"].shape[-2:])
        label_size = tuple(x["img"].shape[-2:])

        #x_backbone = x["layer5"]
        x_backbone = self.block4(x["layer4"])
        # x_backbone = self.block5(x["layer4"], backbone=x_backbone)
        # x_backbone = self.block6(x["layer4"], backbone=x_backbone)
        # x_backbone = self.block7(x["layer4"], backbone=x_backbone)

        x_aspp = self.aspp(x_backbone)
        x_aspp = nn.Upsample(l2_size, mode='bilinear', align_corners=True)(x_aspp)
        x = torch.cat((self.low_level_feature_reducer(x["layer2"]), x_aspp), dim=1)
        x = self.decoder(x)
        x = nn.Upsample(label_size, mode='bilinear', align_corners=True)(x)
        return x

    def eval(self):
        self.block4.eval()
        self.aspp.eval()
        self.decoder.eval()
        return self

    def train(self, mode=True):
        self.block4.eval()
        self.aspp.train(mode)
        self.decoder.train(mode)
        return self


class ASPP(nn.Module):

    def __init__(self, C, depth, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth

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
        return x


class CascadeBlock(nn.Module):

    def __init__(self, block, planes, inplanes, blocks, stride=1, dilation=1):
        super(CascadeBlock, self).__init__()
        self.conv = nn.Conv2d
        # downsample = None
        # if stride != 1 or dilation != 1 or inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         self.conv(inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=stride, dilation=max(1, dilation // 2), bias=False),
        #         self._make_norm(planes * block.expansion),
        #     )
        #
        # layers = []
        # self.upsample_layer = block(inplanes, planes, stride, downsample, dilation=max(1, dilation // 2),
        #                         conv=self.conv, norm=self._make_norm)
        # inplanes = planes * block.expansion
        # for i in range(1, blocks):
        #     layers.append(block(inplanes, planes, dilation=dilation, conv=self.conv, norm=self._make_norm))
        # self.conv = nn.Sequential(*layers)

        downsample = nn.Sequential(
            self.conv(inplanes, planes*block.expansion, kernel_size=1, stride=stride,
                      dilation=dilation, bias=False),
            self._make_norm(planes * block.expansion),
        )
        self.upsample_layer = block(inplanes, planes, stride, downsample, dilation=dilation,
                                    conv=self.conv, norm=self._make_norm)
        inplanes = planes * block.expansion
        self.conv = nn.Sequential(
            block(inplanes, planes, dilation=dilation*2, conv=self.conv, norm=self._make_norm),
            block(inplanes, planes, dilation=dilation, conv=self.conv, norm=self._make_norm)
        )

    def forward(self, x, backbone=None):
        out = self.upsample_layer(x)
        if backbone is not None:
            out = out + backbone
        out = self.conv(out)
        return out

    def _make_norm(self, planes, momentum=0.05):
        return nn.BatchNorm2d(planes, momentum=momentum)
