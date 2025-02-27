import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                            kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 4*growth_rate, 1, 1, 0)
        self.conv2 = ConvBlock(4*growth_rate, growth_rate, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([x, out], dim=1)
    

class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, in_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        for i in range(num_layers):
            self.layers.add_module(f'denselayer{i+1}', DenseLayer(self.in_channels, growth_rate))
            self.in_channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))
    

class DenseNet(nn.Module):
    def __init__(self, blocks_cfg, in_channels, num_classes, growth_rate=32, compression=0.5):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        in_channels = 64
        for i, num_layers in enumerate(blocks_cfg):
            self.features.add_module(f'denseblock{i+1}', DenseBlock(num_layers, growth_rate, in_channels))
            in_channels += num_layers * growth_rate

            if i != len(blocks_cfg ) - 1:
                out_channels = int(in_channels * compression)
                self.features.add_module(f'transition{i+1}', TransitionLayer(in_channels, out_channels))
                in_channels = out_channels
        self.features.add_module('norm5', nn.BatchNorm2d(in_channels))
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, start_dim=1)
        out = self.head(out)
        return out
    

def densenet121(**kwargs):
    return DenseNet([6, 12, 24, 16], in_channels=3, **kwargs)


class DenseNetModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = densenet121(num_classes=n_classes)

    def forward(self, x):
        return self.model(x)
    

