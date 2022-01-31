#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .aspp import ASPP
from .decoder import Decoder

n_class = 41


class Deeplab(nn.Module):
    def __init__(self, fine_tune=False, depth=False, resnext=False):
        super(Deeplab, self).__init__()
        self.encoder = Encoder(fine_tune, resnext=resnext)
        self.aspp = ASPP(2048, 256)
        self.decoder = Decoder(n_class, 256, 256) if not depth else Decoder(1, 256, 256)

    def forward(self, x):
        input_res = x.shape[-2:]
        x, low_level_feat = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        out = F.interpolate(x, size=input_res, mode="bilinear", align_corners=True)
        return torch.squeeze(out)


#%%
if __name__ == "__main__":
    model = Deeplab()
    x = torch.randn((2, 3, 192, 256))
    print(model(x).shape)
#%%
