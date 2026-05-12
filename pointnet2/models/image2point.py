import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dec_net import Decoder_Network
from models.diffusion_unet import DiffusionUnet

class Image2Point(nn.Module):
    def __init__(self, img_ch=3, embed_dim=256):
        super(Image2Point, self).__init__()
        self.unet = DiffusionUnet(in_channels=img_ch)
        self.decoder=Decoder_Network(K1=embed_dim,K2=embed_dim,N=embed_dim)

    def forward(self, x, pc):
        x = self.unet(x)
        image_features = x.permute(0, 2, 3, 1)
        image_features = image_features.flatten(1,2)
        x = self.decoder(image_features, pc)
        # x = x.view(x.shape[0], -1, 3)
        return x, image_features