#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.freezer import freeze


class SingleDecoder(nn.Module):
    def __init__(self, num_classes, in_channels, low_level_in_channels):
        super(SingleDecoder, self).__init__()
        self.low_level_block = nn.Sequential(
            nn.Conv2d(low_level_in_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.seg_classifier = nn.Sequential(nn.Conv2d(256, num_classes, 1))
        self.depth_classifier = nn.Sequential(nn.Conv2d(256, 1, 1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_block(low_level_feat)
        # upsample 4x
        x = F.interpolate(
            x, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=True
        )
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.conv_block(x)

        return self.seg_classifier(x), self.depth_classifier(x)


class SharedDecoder(nn.Module):
    def __init__(self, num_classes, in_channels, low_level_in_channels, fine_tune_decoder=False, transpose=False):
        super(SharedDecoder, self).__init__()
        self.low_level_block = nn.Sequential(
            nn.Conv2d(low_level_in_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.seg_feat = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.depth_feat = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.seg_to_dep = nn.Sequential(
            nn.Conv2d(256, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.shared_to_dep = nn.Sequential(
            nn.Conv2d(256, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.depth_classifier = nn.Sequential(
            nn.Conv2d(64 + 1, 1, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.dep_to_seg = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.shared_to_seg = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dep_rep = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.seg_classifier = nn.Sequential(
            nn.Conv2d(256 + 32, num_classes, 3, padding=1),
        )
        
        self.transpose = transpose
        if self.transpose:
            self.deconv1 = nn.ConvTranspose2d(
                256, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
            self.deconv2 = nn.ConvTranspose2d(
                256, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)

        if fine_tune_decoder:
            print("freezing")
            blocks = [self.seg_feat, self.low_level_block, self.dep_rep, self.shared_to_dep, self.shared_to_seg, self.dep_to_seg, self.depth_feat,self.seg_to_dep]
            for block in blocks:
                for layer in block.children():
                    freeze(layer)


    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_block(low_level_feat)
        # upsample 4x
        if self.transpose:
            shared_rep = self.deconv2(self.deconv1(x))
        else:
            shared_rep = F.interpolate(
                x, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=True
            )
        #         print ('shared_rep: ', torch.any(shared_rep.isnan()))
        x = torch.cat((shared_rep, low_level_feat), dim=1)
        #         print ('x: ', torch.any(x.isnan()))
        seg_feat, depth_feat = self.seg_feat(x), self.depth_feat(x)
        #         print ('seg_feat: ', torch.any(seg_feat.isnan()))
        #         print ('depth_feat: ', torch.any(depth_feat.isnan()))

        eps = 1.0e-8
        seg_rep = torch.sqrt(
            self.seg_to_dep(seg_feat) * self.shared_to_dep(shared_rep) + eps
        )
        depth = self.depth_classifier(torch.cat((seg_rep, depth_feat), dim=1))
        #         print ('seg_rep: ', torch.any(seg_rep.isnan()))
        #         print ('depth: ', torch.any(depth.isnan()))

        depth_rep = self.dep_rep(
            self.dep_to_seg(depth_feat) * self.shared_to_seg(shared_rep)
        )
        seg = self.seg_classifier(torch.cat((depth_rep, seg_feat), dim=1))
        #         print ('depth_rep: ', torch.any(depth_rep.isnan()))
        #         print ('seg: ', torch.any(seg.isnan()))

        return seg, depth


class SharedDecoderRecon(nn.Module):
    def __init__(self, num_classes, in_channels, low_level_in_channels, fine_tune_decoder=False, transpose=False):
        super(SharedDecoderRecon, self).__init__()
        self.low_level_block = nn.Sequential(
            nn.Conv2d(low_level_in_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.seg_feat = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.depth_feat = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.shared_feat = nn.Sequential(
            nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.seg_to_dep = nn.Sequential(
            nn.Conv2d(256, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.shared_to_dep = nn.Sequential(
            nn.Conv2d(256, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.depth_classifier = nn.Sequential(
            nn.Conv2d(64 + 1, 1, 5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.recon_classifier = nn.Sequential(
            nn.Conv2d(64, 3, 5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.dep_to_seg = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.shared_to_seg = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dep_rep = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.seg_classifier = nn.Sequential(
            nn.Conv2d(256 + 32, num_classes, 3, padding=1),
        )
        
        self.transpose = transpose
        if self.transpose:
            self.deconv1 = nn.ConvTranspose2d(
                256, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
            self.deconv2 = nn.ConvTranspose2d(
                256, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)

        if fine_tune_decoder:
            print("freezing")
            blocks = [self.seg_feat, self.low_level_block, self.dep_rep, self.shared_to_dep, self.shared_to_seg, self.dep_to_seg, self.depth_feat,self.seg_to_dep]
            for block in blocks:
                for layer in block.children():
                    freeze(layer)


    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_block(low_level_feat)
        # upsample 4x
        if self.transpose:
            shared_rep = self.deconv2(self.deconv1(x))
        else:
            shared_rep = F.interpolate(
                x, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=True
            )
        #         print ('shared_rep: ', torch.any(shared_rep.isnan()))
        x = torch.cat((shared_rep, low_level_feat), dim=1)
        #         print ('x: ', torch.any(x.isnan()))
        seg_feat, depth_feat, shared_feat = self.seg_feat(x), self.depth_feat(x), self.shared_feat(x)
        #         print ('seg_feat: ', torch.any(seg_feat.isnan()))
        #         print ('depth_feat: ', torch.any(depth_feat.isnan()))
        # print ('shared_feat: ', torch.any(shared_feat.isnan()), shared_feat.shape)

        eps = 1.0e-8
        seg_rep = torch.sqrt(
            self.seg_to_dep(seg_feat) * self.shared_to_dep(shared_rep) + eps
        )
        depth = self.depth_classifier(torch.cat((seg_rep, depth_feat), dim=1))
        #         print ('seg_rep: ', torch.any(seg_rep.isnan()))
        #         print ('depth: ', torch.any(depth.isnan()))

        depth_rep = self.dep_rep(
            self.dep_to_seg(depth_feat) * self.shared_to_seg(shared_rep)
        )
        seg = self.seg_classifier(torch.cat((depth_rep, seg_feat), dim=1))
        #         print ('depth_rep: ', torch.any(depth_rep.isnan()))
        #         print ('seg: ', torch.any(seg.isnan()))
        recon = self.recon_classifier(shared_feat)
        return seg, depth, recon