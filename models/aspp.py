#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append('..')
from models.freezer import freeze

# Atrous Spatial Pyramid Pooling
class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, fine_tune_decoder=False):
        def asppBlock(in_channels, out_channels, kernel_size, padding, dilation):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        super(ASPP, self).__init__()
        self.aspp1 = asppBlock(in_channels, out_channels, 1, 0, 1)
        self.aspp2 = asppBlock(in_channels, out_channels, 3, 6, 6)
        self.aspp3 = asppBlock(in_channels, out_channels, 3, 12, 12)
        self.aspp4 = asppBlock(in_channels, out_channels, 3, 18, 18)
        self.aspp5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.combine = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        if fine_tune_decoder:
            print("freezing")
            blocks = [self.aspp1, self.aspp2, self.aspp3, self.aspp4, self.aspp5]
            for block in blocks:
                for param in block.children():
                    freeze(param)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # pooling layer
        x5 = self.aspp5(x)
        # upsample
        x5 = F.interpolate(x5, size=x.shape[-2:], mode="bilinear", align_corners=True)
        # dim 1 is channel, stack all the channels
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.combine(x)


#%%
if __name__ == "__main__":
    aspp = ASPP(2048, 256)
    x = torch.randn((2, 2048, 32, 32))
    out = aspp(x)
    print(out.shape)

#%%
