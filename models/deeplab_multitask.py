#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .aspp import ASPP
from .decoder import Decoder

n_class = 41


class DeeplabMultiTask(nn.Module):
    def __init__(self, fine_tune=False, fine_tune_all=False, resnext=False):
        super(DeeplabMultiTask, self).__init__()
        self.encoder = Encoder(
            fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext
        )
        self.aspp = ASPP(2048, 256)
        self.seg_decoder = Decoder(n_class, 256, 256)
        self.depth_decoder = Decoder(1, 256, 256)
        self.params = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, x):
        input_res = x.shape[-2:]
        x, low_level_feat = self.encoder(x)
        x = self.aspp(x)
        seg = self.seg_decoder(x, low_level_feat)
        depth = self.depth_decoder(x, low_level_feat)
        out_seg = F.interpolate(
            seg, size=input_res, mode="bilinear", align_corners=True
        )
        out_depth = F.interpolate(
            depth, size=input_res, mode="bilinear", align_corners=True
        )
        return torch.squeeze(out_seg), torch.squeeze(out_depth)


#%%
