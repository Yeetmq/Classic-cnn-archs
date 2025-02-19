import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Conv_block(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class Inception_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 num_1x1_channels, 
                 num_3x3_channels, 
                 num_5x5_channels, 
                 num_channels_reduce_3x3, 
                 num_channels_reduce_5x5, 
                 pooling):
        super().__init__()
        
        self.Block_1 = nn.Sequential(
            Conv_block(
                in_channels=in_channels,
                out_channels=num_1x1_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        
        self.Block_2 = nn.Sequential(
            Conv_block(
                in_channels=in_channels,
                out_channels=num_channels_reduce_3x3,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            Conv_block(
                in_channels=num_channels_reduce_3x3,
                out_channels=num_3x3_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.Block_3 = nn.Sequential(
            Conv_block(
                in_channels=in_channels,
                out_channels=num_channels_reduce_5x5,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            Conv_block(
                in_channels=num_channels_reduce_5x5,
                out_channels=num_5x5_channels,
                kernel_size=5,
                stride=1,
                padding=2
            )
        )

        self.Block_4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            Conv_block(
                in_channels=in_channels,
                out_channels=pooling,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, tensor):
        first_block_out = self.Block_1(tensor)
        second_block_out = self.Block_2(tensor)
        third_block_out = self.Block_3(tensor)
        fourth_block_out = self.Block_4(tensor)

        output = torch.concatenate([first_block_out, second_block_out, third_block_out, fourth_block_out], dim=1)
        return output


class Auxiliary_classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.Adaptive_AvgPool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.conv_block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.act = nn.ReLU()
        self.FC_1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.DropOut = nn.Dropout(p=0.7)
        self.FC_2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, tensor):
        tensor = self.Adaptive_AvgPool(tensor)
        tensor = self.conv_block(tensor)
        tensor = self.act(tensor)
        tensor = torch.flatten(tensor, start_dim=1)
        tensor = self.FC_1(tensor)
        tensor = self.DropOut(tensor)
        tensor = self.FC_2(tensor)
        return tensor


class InceptionNet_V1(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.Conv_1 = Conv_block(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.Conv_2 = Conv_block(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.Conv_3 = Conv_block(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.Inception_3a = Inception_block(
            in_channels=192,
            num_1x1_channels=64,
            num_3x3_channels=128,
            num_5x5_channels=32,
            num_channels_reduce_3x3=96,
            num_channels_reduce_5x5=16,
            pooling=32
        )
        self.Inception_3b = Inception_block(
            in_channels=64+128+32+32,
            num_1x1_channels=128,
            num_3x3_channels=192,
            num_5x5_channels=96,
            num_channels_reduce_3x3=128,
            num_channels_reduce_5x5=32,
            pooling=64
        )
        self.MaxPool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.Inception_4a = Inception_block(
            in_channels=128+192+96+64,
            num_1x1_channels=192,
            num_3x3_channels=208,
            num_5x5_channels=48,
            num_channels_reduce_3x3=96,
            num_channels_reduce_5x5=16,
            pooling=64
        )
        self.Inception_4b = Inception_block(
            in_channels=192+208+48+64,
            num_1x1_channels=160,
            num_3x3_channels=224,
            num_5x5_channels=64,
            num_channels_reduce_3x3=112,
            num_channels_reduce_5x5=24,
            pooling=64
        )
        self.Inception_4c = Inception_block(
            in_channels=160+224+64+64,
            num_1x1_channels=128,
            num_3x3_channels=256,
            num_5x5_channels=64,
            num_channels_reduce_3x3=128,
            num_channels_reduce_5x5=24,
            pooling=64
        )
        self.Inception_4c = Inception_block(
            in_channels=160+224+64+64,
            num_1x1_channels=128,
            num_3x3_channels=256,
            num_5x5_channels=64,
            num_channels_reduce_3x3=128,
            num_channels_reduce_5x5=24,
            pooling=64
        )
        self.Inception_4d = Inception_block(
            in_channels=128+256+64+64,
            num_1x1_channels=112,
            num_3x3_channels=288,
            num_5x5_channels=64,
            num_channels_reduce_3x3=144,
            num_channels_reduce_5x5=32,
            pooling=64
        )
        self.Inception_4e = Inception_block(
            in_channels=112+288+64+64,
            num_1x1_channels=256,
            num_3x3_channels=320,
            num_5x5_channels=128,
            num_channels_reduce_3x3=16,
            num_channels_reduce_5x5=32,
            pooling=128
        )

        self.MaxPool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.Inception_5a = Inception_block(
            in_channels=256+320+128+128,
            num_1x1_channels=256,
            num_3x3_channels=320,
            num_5x5_channels=128,
            num_channels_reduce_3x3=16,
            num_channels_reduce_5x5=32,
            pooling=128
        )
        self.Inception_5b = Inception_block(
            in_channels=256+320+128+128,
            num_1x1_channels=384,
            num_3x3_channels=384,
            num_5x5_channels=128,
            num_channels_reduce_3x3=19,
            num_channels_reduce_5x5=48,
            pooling=128
        )
        self.Average_pooling_1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.DropOut = nn.Dropout(p=0.4)
        self.FC = nn.Linear(in_features=1024, out_features=num_classes)
        
        self.Auxiliary_4a = Auxiliary_classifier(in_channels=512, num_classes=num_classes)
        self.Auxiliary_4d = Auxiliary_classifier(in_channels=528, num_classes=num_classes)

    def forward(self, tensor):
        tensor = self.Conv_1(tensor)
        tensor = self.MaxPool_1(tensor)
        tensor = self.Conv_2(tensor)
        tensor = self.Conv_3(tensor)
        tensor = self.MaxPool_2(tensor)
        tensor = self.Inception_3a(tensor)
        tensor = self.Inception_3b(tensor)
        tensor = self.MaxPool_3(tensor)
        tensor = self.Inception_4a(tensor)
        aux1 = self.Auxiliary_4a(tensor)
        tensor = self.Inception_4b(tensor)
        tensor = self.Inception_4c(tensor)
        tensor = self.Inception_4d(tensor)
        aux2 = self.Auxiliary_4d(tensor)
        tensor = self.Inception_4e(tensor)
        tensor = self.MaxPool_4(tensor)
        tensor = self.Inception_5a(tensor)
        tensor = self.Inception_5b(tensor)
        tensor = self.Average_pooling_1(tensor)
        tensor = torch.flatten(tensor, 1)
        tensor = self.DropOut(tensor)
        tensor = self.FC(tensor)
        return tensor, aux1, aux2