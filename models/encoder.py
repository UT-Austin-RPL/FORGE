import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.fusion import ConvGRU_3D


class Encoder3D(nn.Module):
    def __init__(self, config):
        super(Encoder3D, self).__init__()

        # feature extraction
        self.feature_extraction = get_resnet50()

        # predict feature 3D
        self.features_head = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
        )

        # predict density
        self.density_head = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
        )

        # fusion
        self.fusion_feature = ConvGRU_3D(config, n_layers=1, input_size=128, hidden_size=128)


    def get_feat3D(self, img):
        z_2d = self.feature_extraction(img)
        B,C,H,W = z_2d.shape  # downsample 16x
        z_3d = z_2d.view(-1, 64, 32, H, W)     # [B,64,32,32,32]
        z_3d = self.conv1(z_3d)      
        return z_3d

    def get_density3D(self, z_3d):
        return self.density_head(z_3d)

    def get_render_features(self, x):
        return self.features_head(x)

    def fuse(self, x):
        # x in [b,t,c,d,h,w]
        b,t,c,d,h,w = x.shape
        x = self.fusion_feature(x, [self.fusion_feature.fusion_conv(x.mean(dim=1))])
        return x

    def forward(self, x):
        '''Not Implemented, a dummy function'''
        raise NotImplementedError
        return x


def get_resnet50():
    model = torchvision.models.resnet50(pretrained=True)   # weights=ResNet50_Weights.IMAGENET1K_V1
    feature = nn.Sequential(*list(model.children())[:-2])
    feature[7][0].conv2.stride = (1, 1)
    feature[7][0].downsample[0].stride = (1, 1)
    feature[6][0].conv2.stride = (1, 1)
    feature[6][0].downsample[0].stride = (1, 1)
    return feature
