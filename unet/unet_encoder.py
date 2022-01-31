import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_Encoder(nn.Module):
    def __init__(self, number_channels):
        '''puts together the different down and upscale building blocks for Unet together, followed by the last convolutional block
        :param number_channels number of input channels
        :param number_class number of classes to be predicted'''
        super(UNet_Encoder, self).__init__()
        self.number_channels = number_channels

        self.inc = DoubleConv(number_channels, 64)
        self.downsc_1 = DownScale(64, 128)
        self.downsc_2 = DownScale(128, 256)
        self.downsc_3 = DownScale(256, 512)
        self.downsc_4 = DownScale(512, 1024 // 2)

    def forward(self, x):
        x_1 = self.inc(x)
        x_2 = self.downsc_1(x_1)
        x_3 = self.downsc_2(x_2)
        x_4 = self.downsc_3(x_3)
        x_5 = self.downsc_4(x_4)
        return x_5, x_4, x_3, x_2, x_1


class DoubleConv(nn.Module):
    '''Double convolution with kernel 3 batch normalization and relu'''
    def __init__(self, input_channel, output_channels, middle_channels=None):
        super().__init__()
        if not middle_channels:
            middle_channels = output_channels#if we don't have middle channels
        self.double_convolution = nn.Sequential(
            nn.Conv2d(input_channel, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_convolution(x)


class DownScale(nn.Module):
    '''downscale building block of unet'''
    def __init__(self, input_channel, output_channels):
        super().__init__()
        self.maxpool_convolution = nn.Sequential(nn.MaxPool2d(2), DoubleConv(input_channel, output_channels))

    def forward(self, x):
        return self.maxpool_convolution(x)

