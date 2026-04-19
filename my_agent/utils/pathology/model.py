import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torchvision.models import resnet34, ResNet34_Weights



# Attention Module

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Residual Block

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# ASPP (Atrous Spatial Pyramid Pooling)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.conv_out = nn.Conv2d(out_ch * 5, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.relu(self.bn(self.conv_out(out)))
        return out


class ResNetUNetAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        
        # Use ResNet34 as encoder
        resnet = models.resnet34(pretrained=pretrained)
        
        # Encoder
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4
        
        # ASPP Bridge
        self.aspp = ASPP(512, 512)
        
        # Decoder with attention
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.dec4 = ResidualBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.dec3 = ResidualBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = ResidualBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.att1 = AttentionBlock(64, 64, 32)
        self.dec1 = ResidualBlock(128, 64)
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )
        
        # Deep supervision outputs (only used during training)
        self.ds4 = nn.Conv2d(256, out_channels, 1)
        self.ds3 = nn.Conv2d(128, out_channels, 1)
        self.ds2 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        e0 = self.enc0(x)  # 64, H/2, W/2
        e1 = self.enc1(e0)  # 64, H/4, W/4
        e2 = self.enc2(e1)  # 128, H/8, W/8
        e3 = self.enc3(e2)  # 256, H/16, W/16
        e4 = self.enc4(e3)  # 512, H/32, W/32
        
        # Bridge with ASPP
        b = self.aspp(e4)
        
        # Decoder with attention
        d4 = self.up4(b)
        e3_att = self.att4(d4, e3)
        d4 = self.dec4(torch.cat([d4, e3_att], dim=1))
        
        d3 = self.up3(d4)
        e2_att = self.att3(d3, e2)
        d3 = self.dec3(torch.cat([d3, e2_att], dim=1))
        
        d2 = self.up2(d3)
        e1_att = self.att2(d2, e1)
        d2 = self.dec2(torch.cat([d2, e1_att], dim=1))
        
        d1 = self.up1(d2)
        e0_att = self.att1(d1, e0)
        d1 = self.dec1(torch.cat([d1, e0_att], dim=1))
        
        # Main output - upsample to input size if needed
        out = self.final_conv(d1)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        # Deep supervision (optional, for training only)
        if self.training:
            ds4 = F.interpolate(self.ds4(d4), size=input_size, mode='bilinear', align_corners=False)
            ds3 = F.interpolate(self.ds3(d3), size=input_size, mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=input_size, mode='bilinear', align_corners=False)
            return out, ds4, ds3, ds2
        
        return out
