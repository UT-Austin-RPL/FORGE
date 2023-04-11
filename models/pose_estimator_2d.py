import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.model_utils import CrossAttention, SelfAttention
from einops import rearrange
from models import model_utils


class PoseEstimator2D(nn.Module):
    def __init__(self):
        super(PoseEstimator2D, self).__init__()

        self.backbone = FPN()

        self.cross_attn_layers = 3
        self.self_attn_layers = 3

        self.cross_attn_blks = nn.ModuleList([
            CrossAttention(num_heads=4, num_q_input_channels=256, num_kv_input_channels=256,mlp_ratio=4)
            for i in range(self.cross_attn_layers)
        ])
        self.self_attn_blks = nn.ModuleList([
            SelfAttention(num_heads=4, num_channels=256, mlp_ratio=4)
            for i in range(self.self_attn_layers)
        ])

        self.conv = nn.Sequential(*[
            nn.Conv2d(256, 256, 3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        ])

        self.out = nn.Sequential(*[
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 7)
        ])

        pos_emb = 0.05 * torch.from_numpy(model_utils.get_2d_sincos_pos_embed(256, 16, cls_token=False))
        self.pos_emb = nn.Parameter(pos_emb.unsqueeze(0))


    def forward(self, x, return_features=False):
        '''
        x in shape [B,T,C,H,W] 2D images
        '''
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        feat = self.backbone(x)     # [B*T,256,16,16]
        H2, W2 = feat.shape[-2:]
        feat = rearrange(feat, '(b t) c h w -> b t c h w', t=T)     # [b,t,256,16,16]

        feat_canonical = feat[:,0]  # [b,256,16,16]
        feat = feat[:,1:]           # [b,t-1,256,16,16]

        feat_canonical = rearrange(feat_canonical, 'b c h w -> b (h w) c')     # [b,16*16,256]
        feat = rearrange(feat, 'b t c h w -> b t (h w) c')      # [b,t-1,16*16,256]
        feat_canonical += self.pos_emb.to(feat.device)
        feat += self.pos_emb.unsqueeze(1).to(feat.device)
        feat = rearrange(feat, 'b t n c -> b (t n) c')          # [b, (t-1)*16*16,256]

        for cross_attn_blk, self_attn_blk in zip(self.cross_attn_blks, self.self_attn_blks):
            feat = cross_attn_blk(x_q=feat, x_k=feat_canonical, x_v=feat_canonical, residual=feat)
            feat = self_attn_blk(feat)

        feat = rearrange(feat, 'b (t n) c -> b t c n', t=T-1)   # [b,t-1,256,16*16]
        feat = rearrange(feat, 'b t c (h w) -> (b t) c h w', h=H2, w=W2)  # [b*(t-1),256,16,16]

        feat = self.conv(feat).squeeze()                  # [b*(t-1),1024]
        if return_features:
            return feat
        else:
            pred = self.out(feat)         # [b*(t-1),7]
            #pred = rearrange(pred, '(b t) c -> b t c', t=T-1)
            return pred




class FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(FPN, self).__init__()
        self.in_planes = 64

        resnet = resnet50(pretrained=pretrained)

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.leakyrelu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)     # 
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)    # 32x
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))     # 16x
        # p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)
        return p4 #[p2, p3, p4, p5]


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(3).mean(2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model Encoder"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth"))
    return model


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out