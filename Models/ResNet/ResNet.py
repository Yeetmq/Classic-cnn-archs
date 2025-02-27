import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.functional as F


class Conv_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, tensor):
        return self.act(self.bn(self.conv(tensor)))


class Basic_block(nn.Module):
    expansion = 1
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride=1):
        super().__init__()
        self.conv1 = Conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, tensor):
        out = self.conv1(tensor)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.act(out)
        skip = self.shortcut(tensor)
        out = out + skip
        out = self.act(out)
        return out
    
class BotlleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = Conv_block(in_channels, out_channels, kernel_size=1, stride=stride)
        self.conv2 = Conv_block(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv_block(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.act = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
    def forward(self, tensor):
        out = self.conv1(tensor)
        out = self.conv2(out)
        out = self.conv3(out)
        skip = self.shortcut(tensor)
        out = out + skip
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3):
        super().__init__()
        self.in_planes = 64
        self.act = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(
            out_channels=64,
            block=block,
            num_blocks=num_blocks[0],
            stride=1
        )
        self.layer2 = self._make_layer(
            out_channels=128,
            block=block,
            num_blocks=num_blocks[1],
            stride=2
        )
        self.layer3 = self._make_layer(
            out_channels=256,
            block=block,
            num_blocks=num_blocks[2],
            stride=2
        )
        self.layer4 = self._make_layer(
            out_channels=512,
            block=block,
            num_blocks=num_blocks[3],
            stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, tensor):
        out = self.act(self.bn1(self.conv1(tensor)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        return out

def resnet18(**kwargs):
    return ResNet(Basic_block, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(Basic_block, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(BotlleNeck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(BotlleNeck, [3, 4, 23, 3], **kwargs)

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}

class ResNetModel(nn.Module):
    def __init__(self, model_name, head, n_classes):
        super().__init__()
        model_f, dim_in = model_dict[model_name]
        self.encoder = model_f()

        if head == 'L':
            self.head = nn.Linear(dim_in, n_classes)
        elif head == 'MLP':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=False),
                nn.Linear(dim_in, n_classes)
            )
        else:
            raise NotImplementedError(f'head not supported: {head}')

    def forward(self, tensor):
        out = self.encoder(tensor)
        out = self.head(out)
        return out

