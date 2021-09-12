import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from lib.utils import *
from lib.datasets import *


def load_data(train_path, val_path):
    train_data = sio.loadmat(train_path)
    val_data = sio.loadmat(val_path)
    return train_data, val_data


class Unet(nn.Module):

    def __init__(self, pan_channels, ms_channels, weights_path=None):
        super(Unet, self).__init__()
        channel_input1 = pan_channels + ms_channels
        channel_output1 = channel_input1 * 2
        self.conv1 = Conv(channel_input1, channel_output1)
        channel_input2 = channel_output1 + ms_channels
        self.down1 = Down()
        channel_output2 = channel_input2 * 2
        self.conv2 = Conv(channel_input2, channel_output2)
        channel_input3 = channel_output2 + ms_channels
        self.down2 = Down()
        channel_output3 = channel_input3 * 2
        self.conv3 = Conv(channel_input3, channel_output3)
        channel_input4 = channel_output3 // 4 + channel_output2
        self.up1 = Up()
        channel_output4 = channel_input4 * 2
        self.conv4 = Conv(channel_input4, channel_output4)
        channel_input5 = channel_output4 // 4 + channel_output1
        self.up2 = Up()
        channel_output5 = channel_input5 * 2
        self.conv5 = Conv(channel_input5, channel_output5)
        self.O_conv3 = OutConv(channel_output3, ms_channels)
        self.O_conv4 = OutConv(channel_output4, ms_channels)
        self.O_conv5 = OutConv(channel_output5, ms_channels)

    def forward(self, pan, ms):
        dim = ms.dim() - 3
        ms1 = nn.functional.interpolate(ms, scale_factor=2, mode='nearest')
        ms2 = nn.functional.interpolate(ms, scale_factor=4, mode='nearest')
        # print("pan:", pan.shape)
        # print("ms:", ms.shape)
        # print("ms1:", ms1.shape)
        # print("ms2:", ms2.shape)
        x1 = self.conv1(torch.cat((pan, ms2), dim))
        # print("x1:", x1.shape)
        x2 = self.down1(x1)
        # print("x2:", x2.shape)
        x2 = self.conv2(torch.cat((x2, ms1), dim))
        # print("x2:", x2.shape)
        x3 = self.down2(x2)
        # print("x3:", x3.shape)
        x3 = self.conv3(torch.cat((x3, ms), dim))
        # print("x3:", x3.shape)
        x4 = self.up1(x3)
        # print("x4:", x4.shape)
        x4 = self.conv4(torch.cat((x4, x2), dim))
        # print("x4:", x4.shape)
        x5 = self.up2(x4)
        # print("x5:", x5.shape)
        x5 = self.conv5(torch.cat((x5, x1), dim))
        # print("x5:", x5.shape)
        x3 = self.O_conv3(x3)
        x4 = self.O_conv4(x4)
        x5 = self.O_conv5(x5)

        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        return x3, x4, x5


if __name__ == '__main__':
    device = torch.device('cuda')
    train_data = Datasets('../data/train.mat', device)
    Unet = Unet(1, 8)
    Unet(train_data.gt[0:2, :, :, :], train_data.pan[0:2, :, :, :], train_data.ms[0:2, :, :, :])