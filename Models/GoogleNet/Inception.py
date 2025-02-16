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
        self.Pooling_block = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5,
                stride=3,
            )
        ),
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
        tensor = self.Pooling_block(tensor)
        tensor = self.conv_block(tensor)
        tensor = self.act(tensor)
        tensor = torch.flatten(tensor, dims=1)
        tensor = self.FC_1(tensor)
        tensor = self.DropOut(tensor)
        tensor = self.FC_2(tensor)
        return tensor


class InceptionNet_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = Conv_block(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_2 = Conv_block(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv_3 = Conv_block(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1
        )