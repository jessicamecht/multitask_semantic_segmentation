#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .aspp import ASPP
from .single_decoder import SingleDecoder, SharedDecoder, SharedDecoderRecon

n_class = 41


class DeeplabMultiTaskSingleDecoder(nn.Module):
    def __init__(
        self, fine_tune=False, depth=False, fine_tune_all=False, resnext=False, transpose=False
    ):
        super(DeeplabMultiTaskSingleDecoder, self).__init__()
        self.encoder = Encoder(
            fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext
        )
        self.aspp = ASPP(2048, 256)
        self.decoder = SingleDecoder(n_class, 256, 256)

    def forward(self, x):
        input_res = x.shape[-2:]
        x, low_level_feat = self.encoder(x)
        x = self.aspp(x)
        seg, depth = self.decoder(x, low_level_feat)
        out_seg = F.interpolate(
            seg, size=input_res, mode="bilinear", align_corners=True
        )
        out_depth = F.interpolate(
            depth, size=input_res, mode="bilinear", align_corners=True
        )
        return torch.squeeze(out_seg), torch.squeeze(out_depth)


class DeeplabMultiTaskSharedDecoder(nn.Module):

    def __init__(self, fine_tune=False, depth=False, fine_tune_all=False, resnext=False,\
         fine_tune_decoder=False, transpose=False, reconstruction=False):
        super(DeeplabMultiTaskSharedDecoder, self).__init__()
        self.encoder = Encoder(fine_tune=fine_tune, fine_tune_all=fine_tune_all, resnext=resnext)
        self.aspp = ASPP(2048, 256, fine_tune_decoder)

        self.reconstruction = reconstruction
        if reconstruction:
            self.decoder = SharedDecoderRecon(n_class, 256, 256, fine_tune_decoder, transpose=transpose)
        else:
            self.decoder = SharedDecoder(n_class, 256, 256, fine_tune_decoder, transpose=transpose)
        self.transpose = transpose
        if self.transpose:
            self.deconv_seg1 = nn.ConvTranspose2d(
            n_class, n_class, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
            self.deconv_seg2 = nn.ConvTranspose2d(
                n_class, n_class, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
            self.deconv_depth1 = nn.ConvTranspose2d(
            1, 1, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
            self.deconv_depth2 = nn.ConvTranspose2d(
            1, 1, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)              
            self.deconv_recon1 = nn.ConvTranspose2d(
            3, 3, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
            self.deconv_recon2 = nn.ConvTranspose2d(
            3, 3, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)    
            
    def forward(self, x):
        input_res = x.shape[-2:]
        x, low_level_feat = self.encoder(x)
        x = self.aspp(x)
        if self.reconstruction:
            seg, depth, recon = self.decoder(x, low_level_feat)
            if self.transpose:
                out_seg = self.deconv_seg2(self.deconv_seg1(seg))
                out_depth = self.deconv_depth2(self.deconv_depth1(depth))
                out_recon = self.deconv_recon2(self.deconv_recon1(recon))
            else:
                out_seg = F.interpolate(
                    seg, size=input_res, mode="bilinear", align_corners=True
                )
                out_depth = F.interpolate(
                    depth, size=input_res, mode="bilinear", align_corners=True
                )
                out_recon = F.interpolate(
                    recon, size=input_res, mode="bilinear", align_corners=True
                )
            return torch.squeeze(out_seg), torch.squeeze(out_depth), torch.squeeze(out_recon)
        else:
            seg, depth = self.decoder(x, low_level_feat)

            if self.transpose:
                out_seg = self.deconv_seg2(self.deconv_seg1(seg))
                out_depth = self.deconv_depth2(self.deconv_depth1(depth))
            else:
                out_seg = F.interpolate(
                    seg, size=input_res, mode="bilinear", align_corners=True
                )
                out_depth = F.interpolate(
                    depth, size=input_res, mode="bilinear", align_corners=True
                )
            return torch.squeeze(out_seg), torch.squeeze(out_depth)


#%%
if __name__ == "__main__":
    model = DeeplabMultiTaskSharedDecoder(reconstruction=False, transpose=True)
    x = torch.randn((2, 3, 192, 256))
    print(model(x)[0].shape)
#%%
