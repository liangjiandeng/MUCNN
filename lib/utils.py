import torch
import torch.nn as nn


class Up(nn.Module):

    def __init__(self):
        super().__init__()
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        return self.up(x)


class Down(nn.Module):

    def __init__(self):
        super().__init__()
        self.max_pool_conv = nn.MaxPool2d(2)

    def forward(self, x):
        return self.max_pool_conv(x)


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Loss(nn.Module):
    def __init__(self, gt, gt1, gt2, lr, lr1, lr2):
        super().__init__()
        self.gt = gt
        self.gt1 = gt1
        self.gt2 = gt2
        self.lr = lr
        self.lr1 = lr1
        self.lr2 = lr2

    def forward(self):
        pass
