#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .aspp_shallow import ASPP
from .single_decoder import SingleDecoder, SharedDecoder
n_class = 41


class DeeplabMultiTaskSingleDecoder(nn.Module):
    def __init__(self, fine_tune=False, depth=False, fine_tune_all=False):
        super(DeeplabMultiTaskSingleDecoder, self).__init__()
        self.encoder = Encoder(fine_tune=fine_tune, fine_tune_all=fine_tune_all)
        self.aspp = ASPP(2048, 256)
        self.decoder = SingleDecoder(n_class, 256, 256) 

    def forward(self, x):
        input_res = x.shape[-2:]
        x, low_level_feat = self.encoder(x)
        x = self.aspp(x)
        seg, depth = self.decoder(x, low_level_feat)
        out_seg = F.interpolate(seg, size=input_res, mode="bilinear", align_corners=True)
        out_depth = F.interpolate(depth, size=input_res, mode="bilinear", align_corners=True)
        return torch.squeeze(out_seg), torch.squeeze(out_depth)

    
class DeeplabMultiTaskSharedDecoder(nn.Module):
    def __init__(self, fine_tune=False, depth=False, fine_tune_all=False):
        super(DeeplabMultiTaskSharedDecoder, self).__init__()
        self.encoder = Encoder(fine_tune=fine_tune, fine_tune_all=fine_tune_all)
        self.aspp = ASPP(2048, 256)
        self.decoder = SharedDecoder(n_class, 256, 256) 

    def forward(self, x):
        input_res = x.shape[-2:]
        x, low_level_feat = self.encoder(x)
        x = self.aspp(x)
        seg, depth = self.decoder(x, low_level_feat)
        out_seg = F.interpolate(seg, size=input_res, mode="bilinear", align_corners=True)
        out_depth = F.interpolate(depth, size=input_res, mode="bilinear", align_corners=True)
        return torch.squeeze(out_seg), torch.squeeze(out_depth)

#%%
if __name__ == "__main__":
    model = DeeplabMultiTaskSharedDecoder()
    x = torch.randn((2, 3, 192, 256))
    print(model(x).shape)
#%%
