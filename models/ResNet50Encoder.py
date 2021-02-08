import torch
import torchvision
import torch.nn as nn
import math


class ResNet50Encoder(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        if weights:
            # self.model = torchvision.models.resnet50(pretrained=False)
            self.model = AtrousResNet(Bottleneck, [3, 4, 6, 3, 3, 3, 3])
            weight_dict = torch.load(weights, map_location="cpu")
            for name, weight in list(weight_dict.items()):
                if 'layer4' in name:
                    weight_dict[name.replace("layer4", "layer5")] = weight
                    weight_dict[name.replace("layer4", "layer6")] = weight
                    weight_dict[name.replace("layer4", "layer7")] = weight
            self.load_state_dict(weight_dict, strict=False)
            self.model.layer5 = CascadeBlock(list(self.model.layer5))
            self.model.layer6 = CascadeBlock(list(self.model.layer6))
            self.model.layer7 = CascadeBlock(list(self.model.layer7))
        else:
            self.model = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):

        original_batch_size = x.size(0)
        if len(x.shape) == 5:
            x = x.view(-1, x.size(2), x.size(3), x.size(4))

        res = {}
        res["img"] = x
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
        x4 = self.model.layer4(x)
        x5 = self.model.layer5(x, x_cas=x4)
        x6 = self.model.layer6(x, x_cas=x5)
        x7 = self.model.layer7(x, x_cas=x6)
        x = x7

        res["layer5"] = x



        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # res["embedding"] = x

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


class AtrousResNet(nn.Module):

    def __init__(self, block, layers, num_groups=None, beta=False):
        super(AtrousResNet, self).__init__()
        self.num_groups = num_groups
        self.inplanes = 64
        self.conv = nn.Conv2d
        if not beta:
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self._make_norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        l4_inplanes = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=2)
        self.inplanes = l4_inplanes
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1,
                                       dilation=4)
        self.inplanes = l4_inplanes
        self.layer6 = self._make_layer(block, 512, layers[5], stride=1,
                                       dilation=8)
        self.inplanes = l4_inplanes
        self.layer7 = self._make_layer(block, 512, layers[6], stride=1,
                                       dilation=16)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self._make_norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self._make_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self._make_norm))

        return nn.Sequential(*layers)

    def _make_norm(self, planes, momentum=0.05):
        return nn.BatchNorm2d(planes, momentum=momentum) if self.num_groups is None \
            else nn.GroupNorm(self.num_groups, planes)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x


class CascadeBlock(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.downscale = layers[0]
        self.rest = nn.Sequential(*layers[1:])

    def forward(self, x, x_cas=None):
        out = self.downscale(x)
        if x_cas is not None:
            out += x_cas
        out = self.rest(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
