import timm
import torch
import torch.nn as nn
from functools import partial
from .basic_block_2d import BasicConv2d, BasicDeconv2d


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        # 1. 改进上采样方式：双线性插值+卷积
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(chan_low, chan_high,
                        kernel_size=3, padding=1,
                        norm_layer=nn.BatchNorm2d,
                        act_layer=None)  # 不加激活保证特征线性融合
        )

        # 2. 添加空间注意力机制
        self.attn = nn.Sequential(
            nn.Conv2d(chan_high, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 特征融合卷积
        self.conv = BasicConv2d(chan_high * 2, chan_high,
                                kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        # 上采样低层特征
        low = self.upsample(low)

        # 生成空间注意力掩码
        attn_mask = self.attn(high)

        # 应用注意力机制
        weighted_high = high * (1 + attn_mask)  # 增强重要区域

        # 特征拼接与融合
        fused_feat = torch.cat([weighted_high, low], dim=1)
        return self.conv(fused_feat)


class Backbone(nn.Module):
    def __init__(self, backbone='MobileNetV3'):
        super().__init__()
        if 'v3' in backbone:
            # 初始化主干网络
            self.model = timm.create_model('mobilenetv3_large_100',
                                           pretrained=True,
                                           features_only=True,
                                           out_indices=(0, 1, 2, 3, 4))
            self.channels = self.model.feature_info.channels()

            # 初始化FPN结构
            self.fpn_layer4 = FPNLayer(self.channels[4], self.channels[3])
            self.fpn_layer3 = FPNLayer(self.channels[3], self.channels[2])
            self.fpn_layer2 = FPNLayer(self.channels[2], self.channels[1])
            self.fpn_layer1 = FPNLayer(self.channels[1], self.channels[0])

            # 3. 改进归一化方式并添加残差连接
            self.out_conv1 = nn.Sequential(
                BasicConv2d(self.channels[1], self.channels[1],
                            kernel_size=3, padding=1,
                            padding_mode="replicate",
                            norm_layer=nn.BatchNorm2d),  # 改为BatchNorm
                nn.ReLU(inplace=True)
            )
            self.skip_conv1 = BasicConv2d(self.channels[1], self.channels[1],
                                          kernel_size=1,
                                          norm_layer=None,
                                          act_layer=None)

            self.out_conv = nn.Sequential(
                BasicConv2d(self.channels[0], self.channels[0],
                            kernel_size=3, padding=1,
                            padding_mode="replicate",
                            norm_layer=nn.BatchNorm2d),  # 改为BatchNorm
                nn.ReLU(inplace=True)
            )
            self.skip_conv = BasicConv2d(self.channels[0], self.channels[0],
                                         kernel_size=1,
                                         norm_layer=None,
                                         act_layer=None)

            self.output_channels = [
                self.channels[0], self.channels[1],
                self.channels[2], self.channels[3], self.channels[4]
            ]

    def forward(self, images):
        # 提取多尺度特征
        features = self.model(images)
        c0, c1, c2, c3, c4 = features

        # 特征金字塔融合
        p4 = self.fpn_layer4(c4, c3)
        p3 = self.fpn_layer3(p4, c2)
        p2 = self.fpn_layer2(p3, c1)
        p1 = self.fpn_layer1(p2, c0)

        # 带残差连接的输出处理
        # 处理p2分支
        identity1 = self.skip_conv1(c1)
        p2 = self.out_conv1(p2) + identity1

        # 处理p1分支
        identity0 = self.skip_conv(c0)
        p1 = self.out_conv(p1) + identity0

        return [p1, p2, p3, p4, c4]