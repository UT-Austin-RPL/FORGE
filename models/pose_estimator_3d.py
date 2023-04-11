import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from utils import geo_utils
from models.model_utils import Block, get_3d_sincos_pos_embed


class PoseEstimator3D(nn.Module):
    def __init__(self, config):
        super(PoseEstimator3D, self).__init__()

        self.rot_representation = config.network.rot_representation
        assert self.rot_representation in ['euler', 'quat', '6D', '9D']
        if self.rot_representation == 'euler':
            self.rot_dim = 3
        elif self.rot_representation == 'quat':
            self.rot_dim = 4
        elif self.rot_representation == '6D':
            self.rot_dim = 6
        elif self.rot_representation == '9D':
            self.rot_dim = 9
        self.trans_dim = 3
        self.pose_dim = self.trans_dim + self.rot_dim

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1)
        )
        
        self.coord_dim = 64
        self.pose_transformer = PoseTransformer(inp_res=16, dim=64, mlp_ratio=2, coord_dim=self.coord_dim)
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 512, 3, padding=1, stride=2),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.pose_head_1 = nn.Sequential(*[
            nn.Conv3d(512, 512, 3, padding=1, stride=2),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(512, 1024, 3, padding=1, stride=2),]
        )
        self.pose_head_2 = nn.Sequential(*[
            #nn.BatchNorm3d(1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(inplace=True),]
        )
        self.out = nn.Sequential(*[
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, self.pose_dim+1)
        ])
        

    def forward(self, features, return_features=False):
        # '''
        # features in shape [b,t,C=128,D,H,W]
        # '''
        b,t,C1,D1,H1,W1 = features.shape                                        # spatial size 16
        device = features.device

        features = features.reshape(b*t,C1,D1,H1,W1)
        features = self.conv3d_1(features)                                      # [b*t,C=64,D,H,W]
        _,C,D,H,W = features.shape
        N = D * H * W

        features = features.reshape(b,t,C,N)                                    # [b,t,C,N]
        features_ref = features[:,0:1].repeat(1,t-1,1,1).reshape(b*(t-1),C,N)   # [b*(t-1),C,N]
        features_cur = features[:,1:].reshape(b*(t-1),C,N)

        x = self.pose_transformer(q=features_ref, k=features_cur)               # [b*(t-1),C,N]
        x = x.reshape(b*(t-1), self.coord_dim, D, H, W)
        
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)     # [B,1024,4,4,4]
        # x = self.pose_head(x).view(b*(t-1), self.pose_dim + 1)
        x = self.pose_head_2(self.pose_head_1(x).squeeze())       # [b*(t-1),1024]
        if return_features:
            return x
        else:
            x = self.out(x)             # [b*(t-1),7+1]
            x, conf = x.split([self.pose_dim, 1], dim=-1)
            return x, conf


    def toSE3(self, x):
        if self.rot_representation == 'euler':
            res = geo_utils.euler2mat(x)
        elif self.rot_representation == 'quat':
            res = geo_utils.quat2mat(x)
        elif self.rot_representation == '6D':
            res = geo_utils.rot6d2mat(x)
        elif self.rot_representation == '9D':
            res = geo_utils.rot9d2mat(x)
        return res


class PoseTransformer(nn.Module):
    def __init__(self, inp_res=32, dim=64, mlp_ratio=1, coord_dim=64):
        super().__init__()
        
        self.coord_dim = coord_dim

        self.cross_transformer = Block(dim=dim, mlp_ratio=mlp_ratio, return_attn=True)
        self.self_transformer = Block(dim=dim, mlp_ratio=mlp_ratio, return_attn=False)
        
        # self.penc3d_module = PositionalEncoding3D(channels=self.coord_dim)
        # _ = self.penc3d_module(torch.zeros(1,inp_res,inp_res,inp_res,self.coord_dim))
        # self.pos_embed_3d_coord = self.penc3d_module.cached_penc / 6     # [1,D,H,W,32]
        # del self.penc3d_module
        self.pos_embed_3d_coord = get_3d_sincos_pos_embed(coord_dim, inp_res, inp_res) * 0.1  # [D,H,W,32]

        self.pos_embed_3d_coord = self.pos_embed_3d_coord.reshape(1,-1,self.coord_dim)


    def forward(self, q, k, q_pe=None, k_pe=None):
        '''
        q and k: volume features in shape [B,16,N]
        '''
        B,C,N = q.shape
        pe = self.pos_embed_3d_coord.to(q)                      # [1,N,C]
        attn = self.cross_transformer.get_attn(query=q, key=k)  # [B,N,N]
        coord = torch.matmul(attn, pe)                          # [B,N,C]
        coord = coord.permute(0,2,1)                            # [B,C,N]
        coord = self.self_transformer(query=coord, key=coord)   # [B,C,N]
        return coord