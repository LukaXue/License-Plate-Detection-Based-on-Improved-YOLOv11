"""
轻量级特征融合和特征增强模块实现
这些模块可以集成到YOLO11中，提升模型性能的同时保持轻量级特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LightweightFFM(nn.Module):
    """轻量级特征融合模块 (Lightweight Feature Fusion Module)"""

    def __init__(self, in_channels, out_channels, reduction=16):
        super(LightweightFFM, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 特征压缩
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)

        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa

        return x


class LightweightFEM(nn.Module):
    """轻量级特征增强模块 (Lightweight Feature Enhancement Module)"""

    def __init__(self, channels, expansion=2):
        super(LightweightFEM, self).__init__()
        hidden_channels = channels * expansion

        # 深度可分离卷积
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw_conv1 = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.pw_conv2 = nn.Conv2d(hidden_channels, channels, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU6(inplace=True)

        # 残差连接
        self.use_residual = True

    def forward(self, x):
        identity = x

        # 深度卷积
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第一个逐点卷积（扩展）
        out = self.pw_conv1(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第二个逐点卷积（压缩）
        out = self.pw_conv2(out)
        out = self.bn3(out)

        # 残差连接
        if self.use_residual:
            out = out + identity

        return out




class SimAM(nn.Module):
    """SimAM注意力机制 - 无参数注意力模块"""

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class CAMConv(nn.Module):
    """内容感知卷积 (Content-Aware Convolutional)"""

    def __init__(self, channels, reduction=16):
        super(CAMConv, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # 内容感知模块
        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # 内容感知权重
        weight = self.cam(out)
        out = out * weight

        out = self.relu(out)
        return out


class CrossScaleFusion(nn.Module):
    """跨尺度特征融合模块"""

    def __init__(self):
        super(CrossScaleFusion, self).__init__()

    def forward(self, features: List[torch.Tensor]):
        """
        features: 包含不同尺度特征的列表 [P3, P4, P5]
        """
        p3, p4, p5 = features

        # 获取P3的尺寸作为目标尺寸
        target_size = p3.shape[2:]

        # 将P4和P5上采样到P3的尺寸
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)

        # 简单的特征融合（可以用更复杂的方法）
        fused = p3 + p4_up + p5_up

        return fused


class GhostConv(nn.Module):
    """Ghost卷积 - 减少参数量的轻量级卷积"""

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostConv, self).__init__()
        self.oup = oup
        init_channels = oup // ratio
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class ShuffleAttention(nn.Module):
    """Shuffle Attention模块"""

    def __init__(self, channel, G=8):
        super(ShuffleAttention, self).__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out





if __name__ == "__main__":
    # 测试模块
    x = torch.randn(2, 256, 64, 64)

    # 测试轻量级特征融合模块
    ffm = LightweightFFM(256, 256)
    out_ffm = ffm(x)
    print(f"LightweightFFM输出形状: {out_ffm.shape}")

