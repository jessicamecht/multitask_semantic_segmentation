#%%
import torch
import torch.nn as nn
from torchvision import models


# Reference
# https://arxiv.org/pdf/1802.02611.pdf
class Encoder(nn.Module):
    def __init__(self, fine_tune=False, fine_tune_all=False, resnext=False):
        super(Encoder, self).__init__()
        if resnext:
            resnet = models.resnext101_32x8d(pretrained=True)
        else:
            resnet = models.resnet101(pretrained=True)
        if not fine_tune_all:
            for param in resnet.children():
                param.required_grad = False
        # make the downsampling only 16 times smaller instead of 32
        layers = list(resnet.children())[:-2]
        list(layers[-1][0].children())[2].stride = (1, 1)
        list(layers[-1][0].children())[-1][0].stride = (1, 1)
        self.preprocess = nn.Sequential(*layers[:4])
        self.layer1 = nn.Sequential(*layers[4])
        self.layer23 = nn.Sequential(*layers[5:-1])
        self.layer4 = nn.Sequential(*layers[-1])
        if fine_tune:
            for param in self.layer4.children():
                param.required_grad = True

    def forward(self, x):
        x = self.preprocess(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer23(x)
        out = self.layer4(x)
        return out, low_level_feat


#%%
if __name__ == "__main__":
    resnet = models.resnet101(pretrained=True)
    for param in resnet.children():
        param.required_grad = False
    #%%
    print(list(resnet.children())[-2:])
    #%%
    layers = list(resnet.children())[:-2]
    list(layers[-1][0].children())[2].stride = (1, 1)
    list(layers[-1][0].children())[-1][0].stride = (1, 1)
    #%%
    model = Encoder()
    #%%
    x = torch.randn((2, 3, 32, 32))
    #%%
    out, low_level_feat = model(x)
    #%%
    out.shape
    #%%
    low_level_feat.shape

    #%%
    model = models.resnext101_32x8d(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])
    #%%
    model
    #%%
    x = torch.randn((2, 3, 32, 32))

    #%%
    model(x).shape
    #%%
    len(model.resnet)