import torch
import torch.nn as nn
from unet.unet_encoder import UNet_Encoder
from unet.unet_decoder import UNet_Decoder


class UNet_Multitask(nn.Module):
    def __init__(self, init_channels=3, number_class=41):
        super(UNet_Multitask, self).__init__()
        self.encoder = UNet_Encoder(init_channels)
        self.decoder_task_1 = UNet_Decoder(number_class)
        self.decoder_task_2 = UNet_Decoder(1)

    def forward(self, x):
        x_5, x_4, x_3, x_2, x_1 = self.encoder(x)
        d1 = self.decoder_task_1(x_5, x_4, x_3, x_2, x_1)
        d2 = self.decoder_task_2(x_5, x_4, x_3, x_2, x_1).squeeze()
        return d1, d2
