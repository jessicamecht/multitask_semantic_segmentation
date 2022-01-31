import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_Decoder(nn.Module):
    def __init__(self, number_class):
        '''puts together the different down and upscale building blocks for Unet together, followed by the last convolutional block
        :param number_channels number of input channels
        :param number_class number of classes to be predicted'''
        super(UNet_Decoder, self).__init__()
        self.upsc_1 = UpScale(1024, 512 // 2)
        self.upsc_2 = UpScale(512, 256 // 2)
        self.upsc_3 = UpScale(256, 128 // 2)
        self.upsc_4 = UpScale(128, 64)
        self.outconvolution = OutConv(64, number_class)

    def forward(self, x_5, x_4, x_3, x_2, x_1):
        x = self.upsc_1(x_5, x_4)
        x = self.upsc_2(x, x_3)
        x = self.upsc_3(x, x_2)
        x = self.upsc_4(x, x_1)
        return self.outconvolution(x)

class OutConv(nn.Module):
    '''last conv layer'''
    def __init__(self, input_channel, output_channels):
        super(OutConv, self).__init__()
        self.outconvolution = nn.Conv2d(input_channel, output_channels, kernel_size=1)

    def forward(self, x):
        return self.outconvolution(x)

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


class UpScale(nn.Module):
    '''upscale building block of unet, upsamples by a factor of 2 with bilinear mode'''
    def __init__(self, input_channel, output_channels):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convolution = DoubleConv(input_channel, output_channels, input_channel // 2)

    def forward(self, x_1, x_2):
        x_1 = self.up_sample(x_1)

        difference_Y, difference_X = x_2.size()[2] - x_1.size()[2], x_2.size()[3] - x_1.size()[3]
        x_1 = F.pad(x_1, [difference_X // 2, difference_X - difference_X // 2, difference_Y // 2, difference_Y - difference_Y // 2])#pads the tensor [padding_left,padding_right, padding_top, padding_bottom]
        res = torch.cat([x_2, x_1], dim=1)#concatenate
        return self.convolution(res)

