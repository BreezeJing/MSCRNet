from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from .deformable_refine import DeformableRefine, DeformableRefineF
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .smod import *
from .affinity_feature import *
from .mobilenetv2 import Backbone, FPNLayer
from .basic_block_2d import BasicConv2d
from models.U_net import U_Net, U_Net_F, U_Net_F_v2

class LearnableUpsample(nn.Module):
    """
    可学习的上采样模块（使用转置卷积）
    您可以根据实际需要调整通道数、卷积核大小和步幅。
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(LearnableUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in_channels, out_channels,
                        kernel_size=3, padding=1,
                        norm_layer=nn.BatchNorm2d,
                        act_layer=None)  # 不加激活保证特征线性融合
        )

    def forward(self, x):
        return self.up(x)

class SELayer2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class refinenet_version3(nn.Module):
    def __init__(self, in_channels):
        super(refinenet_version3, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            convbn1(in_channels, 256, 3, 1, 1, 1),
            Mish())

        self.conv2 = nn.Sequential(
            convbn1(256, 128, 3, 1, 1, 1),
            Mish())
        self.conv3 = nn.Sequential(
            convbn1(128, 64, 3, 1, 2, 2),
            Mish())
        self.conv4 = nn.Sequential(
            convbn1(64, 64, 3, 1, 4, 4),
            Mish())
        self.conv5 = self._make_layer(BasicBlock, 64, 1, 1, 1, 8)
        self.conv6 = self._make_layer(BasicBlock, 64, 1, 1, 1, 16)
        self.conv7 = self._make_layer(BasicBlock, 32, 1, 1, 1, 1)

        self.conv8 = nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

        self.se1 = SELayer1(32, reduction=16)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x, disp):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        se1=self.se1(conv7)
        conv8 = self.conv8(se1)
        disp = disp + conv8
        return disp
class refinenet_version1(nn.Module):
    def __init__(self, in_channels):
        super(refinenet_version1, self).__init__()

        self.inplanes = 32
        self.conv1 = nn.Sequential(
            convbn1(in_channels, 128, 3, 1, 1, 1),
            Mish())

        self.conv2 = nn.Sequential(
            convbn1(128, 64, 3, 1, 1, 1),
            Mish())
        self.conv3 = nn.Sequential(
            convbn1(64, 32, 3, 1, 2, 2),
            Mish())
        self.conv4 = nn.Sequential(
            convbn1(32, 32, 3, 1, 4, 4),
            Mish())
        self.conv5 = self._make_layer(BasicBlock, 32, 1, 1, 1, 8)
        self.conv6 = self._make_layer(BasicBlock, 16, 1, 1, 1, 16)
        self.conv7 = self._make_layer(BasicBlock, 16, 1, 1, 1, 1)

        self.conv8 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False)

        self.se2 = SELayer2(16, reduction=16)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x, disp):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        se2=self.se2(conv7)
        conv8 = self.conv8(se2)
        disp = disp + conv8
        return disp
class SELayer1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

class MYNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume, struct_fea_c, fuse_mode, affinity_settings, udc):
        super(MYNet, self).__init__()
        self.num_groups = 8
        self.maxdisp = maxdisp
        self.sfc = struct_fea_c
        self.affinity_settings = affinity_settings
        self.udc = udc
        # backbobe
        self.backbone = Backbone('MobileNetv2')

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = hourglass(32)
        # self.dres3 = hourglass(32)
        # self.dres4 = hourglass(32)

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.use_concat_volume = use_concat_volume
        self.concat_channels = 12

        # 学习型上采样模块，用于特征的上下采样
        self.feature_upsample_to_half = LearnableUpsample(in_channels=32, out_channels=32, scale_factor=2)
        self.feature_upsample_to_full = LearnableUpsample(in_channels=32, out_channels=32, scale_factor=2)


        # 学习型上采样，用于视差图的超分辨率
        self.disp_upsample_1_4_to_1_2 = LearnableUpsample(in_channels=1, out_channels=1, scale_factor=2)
        self.disp_upsample_1_2_to_1_1 = LearnableUpsample(in_channels=1, out_channels=1, scale_factor=2)

        # 视差特征提取
        self.disp_feature_extract = nn.Sequential(
            convbn1(1, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True)
        )

        refinenet_in_channels = 146
        self.refinenet3 = refinenet_version3(refinenet_in_channels)
        self.refinenet1 = refinenet_version1(59)
        self.DispFusion1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.DispFusion2 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]* m.kernel_size[1]* m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0]* m.kernel_size[1]*m.kernel_size[2]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        # 特征提取(1/4分辨率)
        features_left = self.backbone(left)
        features_left_1_4 = features_left[1]
        features_right = self.backbone(right)
        features_right_1_4 =features_right[1]
        cost_volume = build_gwc_volume(features_left_1_4, features_right_1_4, 48, self.num_groups)

        cost0 = self.dres0(cost_volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        cost3 = self.classif3(out1)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparity_regression(pred3, int(self.maxdisp/4))
        disp3_1_4 = torch.unsqueeze(pred3, 1)

        # 将视差从1/4上采样到1/2
        disp3_1_2 = F.interpolate(disp3_1_4 * 2, [int(left.size()[2]/2), int(left.size()[3]/2)], mode='bilinear')
        disp3_1_1 = F.interpolate(disp3_1_4 * 4, [int(left.size()[2]), int(left.size()[3])], mode='bilinear')

        # 将特征从1/4分辨率上采样到1/2分辨率
        # 原特征通道为32，这里直接用feature_upsample_to_half对左、右特征进行上采样
        # features_left_1_2_2 = self.feature_upsample_to_half(features_left_1_4_4)
        # features_right_1_2_2 = self.feature_upsample_to_half(features_right_1_4_4)
        # features_left_1_2 = torch.cat((features_left[0],features_left_1_2_2), dim=1)
        # features_right_1_2 = torch.cat((features_right[0],features_right_1_2_2), dim=1)
        features_left_1_2 = features_left[0]
        features_right_1_2 = features_right[0]
        features_left_1_1 = self.feature_upsample_to_full(features_left_1_2)
        features_right_1_1 = self.feature_upsample_to_full(features_right_1_2)

        # warp右特征(1/2分辨率)
        right_warped_1_2 = warp(features_right_1_2, disp3_1_2)

        # 构建1/2分辨率下的相关性体
        costvolume_1_2 = build_corrleation_volume(features_left_1_2, right_warped_1_2, 24, 1)
        costvolume_1_2 = torch.squeeze(costvolume_1_2, 1)

        disp3_feature = self.disp_feature_extract(disp3_1_2)

        # 拼接: (左特征-右特征warp), 左特征, disp3_1_2, disp3_feature, costvolume_1_2
        refinenet_combine = torch.cat((features_left_1_2 - right_warped_1_2,
                                       features_left_1_2,
                                       disp3_1_2,
                                       disp3_feature,
                                       costvolume_1_2), dim=1)

        # 1/2分辨率下精炼
        disp_refined_1_2 = self.refinenet3(refinenet_combine, disp3_1_2)

        # 将视差从1/2上采样到全分辨率
        disp_full = F.interpolate(disp_refined_1_2 * 2, [int(left.size()[2]), int(left.size()[3])], mode='bilinear')
        # features_left_1_1 = self.feature_upsample_to_half(features_left_1_2)
        # 全分辨率下的最终微调
        right_warped_1_1 = warp(features_right_1_1, disp_full)
        # 构建1/2分辨率下的相关性体
        costvolume_1_1 = build_corrleation_volume(features_left_1_1, right_warped_1_1, 12, 1)
        costvolume_1_1 = torch.squeeze(costvolume_1_1, 1)
        refinenet_combine_1 = torch.cat((features_left_1_1,
                                       disp_full,
                                       disp3_1_1,
                                       costvolume_1_1), dim=1)
        costr = self.refinenet1(refinenet_combine_1, disp_full)

        return disp3_1_1, disp_full, costr
