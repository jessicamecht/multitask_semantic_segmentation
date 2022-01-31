#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append('..')
from models.freezer import freeze


class Decoder(nn.Module):
    def __init__(self, num_classes, in_channels, low_level_in_channels, fine_tune_decoder=False):
        super(Decoder, self).__init__()
        self.low_level_block = nn.Sequential(
            nn.Conv2d(low_level_in_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )
        if fine_tune_decoder:
            for param in self.low_level_block.children():
                freeze(param)

    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_block(low_level_feat)
        # upsample 4x
        x = F.interpolate(
            x, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=True
        )
        x = torch.cat((x, low_level_feat), dim=1)
        return self.classifier(x)

# %%
