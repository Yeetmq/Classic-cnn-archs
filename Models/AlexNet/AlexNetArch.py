import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, in_channel: int, n_classes: int, dropout_val: float = 0.5):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # image shape после conv_1 = 55 x 55 x 96

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # image shape = 27 x 27 x 256

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # image shape = 6 x 6 x 256

        self.fcl1 = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_val)
        )

        self.fcl2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_val)
        )

        self.fcl3 = nn.Sequential(
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fcl1(x)
        x = self.fcl2(x)
        x = self.fcl3(x)
        
        return x
