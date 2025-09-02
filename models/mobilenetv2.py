import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .basic_block_2d import BasicConv2d, BasicDeconv2d
class AffinityFeature(nn.Module):
    def __init__(self, win_h, win_w, dilation, cut):
        super(AffinityFeature, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0

    def padding(self, x, win_h, win_w, dilation):
        pad_t = (win_w // 2 * dilation, win_w // 2 * dilation,
                 win_h // 2 * dilation, win_h // 2 * dilation)
        out = F.pad(x, pad_t, mode='constant')
        return out

    def forward(self, feature):
        B, C, H, W = feature.size()
        feature = F.normalize(feature, dim=1, p=2)
        unfold_feature = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(feature)
        all_neighbor = unfold_feature.reshape(B, C, -1, H, W).transpose(1, 2)
        num = (self.win_h * self.win_w) // 2
        neighbor = torch.cat((all_neighbor[:, :num], all_neighbor[:, num+1:]), dim=1)
        feature = feature.unsqueeze(1)
        affinity = torch.sum(neighbor * feature, dim=2)
        affinity[affinity < self.cut] = self.cut

        return affinity
class StructureFeature(nn.Module):
    def __init__(self, affinity_settings, sfc):
        super(StructureFeature, self).__init__()

        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.sfc = sfc

        in_c = self.win_w * self.win_h - 1

        self.sfc_conv1 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv2 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv3 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv4 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))

    def forward(self, x):

        affinity1 = AffinityFeature(self.win_h, self.win_w, self.dilation[0], 0)(x)
        affinity2 = AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)(x)
        affinity3 = AffinityFeature(self.win_h, self.win_w, self.dilation[2], 0)(x)
        affinity4 = AffinityFeature(self.win_h, self.win_w, self.dilation[3], 0)(x)

        affi_feature1 = self.sfc_conv1(affinity1)
        affi_feature2 = self.sfc_conv2(affinity2)
        affi_feature3 = self.sfc_conv3(affinity3)
        affi_feature4 = self.sfc_conv4(affinity4)

        out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1)
        affinity = torch.cat((affinity1, affinity2, affinity3, affinity4), dim=1)

        # out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3), dim=1)
        # affinity = torch.cat((affinity1, affinity2, affinity3), dim=1)

        return out_feature, affinity
class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.ReLU, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.ReLU, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))
class Backbone(nn.Module):
    def __init__(self, backbone='MobileNetv2'):
        super().__init__()
        if backbone == 'MobileNetv2':
            model = timm.create_model('mobilenetv2_120d', pretrained=False, features_only=True)
            # 根据实际输出调整通道顺序（浅层到深层）
            channels = [384, 112, 40, 32, 24]  # 逆序排列用于FPN
        else:
            raise NotImplementedError
        affinity_settings = {}
        affinity_settings['win_w'] = 3
        affinity_settings['win_h'] = 3
        affinity_settings['dilation'] = [1, 2, 4, 8]
        self.sfc = 4
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        # 根据实际blocks结构调整（假设有7个blocks）
        self.block0 = model.blocks[0]  # 第1个block
        self.block1 = model.blocks[1]  # 第2个block
        self.block2 = model.blocks[2]  # 第3个block
        self.block3 = model.blocks[3:5]  # 第4-5个block（输出112通道）
        self.block4 = model.blocks[5:7]  # 第6-7个block（输出384通道）

        self.embedding_l1 = nn.Sequential(convbn(16, 32, kernel_size=3, stride=1, pad=1, dilation=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))
        self.to_sf_l1 = StructureFeature(affinity_settings, self.sfc)

        self.embedding_l2 = nn.Sequential(convbn(16, 32, kernel_size=3, stride=1, pad=1, dilation=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))
        self.to_sf_l2 = StructureFeature(affinity_settings, self.sfc)

        # 调整FPN通道对应关系
        self.fpn_layer4 = FPNLayer(channels[0], channels[1])  # 384→112
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])  # 112→40
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])  # 40→32
        self.fpn_layer1 = FPNLayer(channels[3], channels[4])  # 32→24

        # 调整输出层通道
        self.out_conv1 = BasicConv2d(channels[4], 16,  # 保持24通道
                                     kernel_size=3, padding=1,
                                     norm_layer=nn.BatchNorm2d)
        self.out_conv = BasicConv2d(32, 16,  # 最终输出通道
                                    kernel_size=3, padding=1,
                                    norm_layer=nn.BatchNorm2d)
        self.output_channels = channels
        self.adjust_s_c_f_1=nn.Sequential(convbn(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))

        self.adjust_s_c_f_2 = nn.Sequential(convbn(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))
    def forward(self, images):
        # 前向传播需要与新的block结构匹配
        c1 = self.act1(self.bn1(self.conv_stem(images)))  # [24, H/2, W/2]
        c1 = self.block0(c1)  # [24, H/2, W/2]
        c2 = self.block1(c1)  # [32, H/4, W/4]
        c3 = self.block2(c2)  # [40, H/8, W/8]
        c4 = self.block3(c3)  # [112, H/16, W/16]
        c5 = self.block4(c4)  # [384, H/32, W/32]


        # FPN特征融合
        p4 = self.fpn_layer4(c5, c4)  # 384→112
        p3 = self.fpn_layer3(p4, c3)  # 112→40
        p2 = self.fpn_layer2(p3, c2)  # 40→32
        p1 = self.fpn_layer1(p2, c1)  # 32→24

        # 最终输出处理
        p1 = self.out_conv1(p1)
        p2 = self.out_conv(p2)
        #结构特征提取
        embedding_l1 = self.embedding_l1(p1)
        l1_sf, l1_affi = self.to_sf_l1(embedding_l1)
        embedding_l2 = self.embedding_l2(p2)
        l2_sf, l2_affi = self.to_sf_l2(embedding_l2)
        p1_s = torch.cat((p1, l1_sf),dim=1)
        p1_s = self.adjust_s_c_f_1(p1_s)
        p2_s = torch.cat((p2, l2_sf), dim=1)
        p2_s = self.adjust_s_c_f_2(p2_s)

        return [p1_s, p2_s, p3, p4, c5]
# class Backbone(nn.Module):
#     def __init__(self, backbone='MobileNetv2'):
#         super().__init__()
#         if backbone == 'MobileNetv2':
#             model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
#             channels = [160, 96, 32, 24, 16]
#         else:
#             raise NotImplementedError
#
#         self.conv_stem = model.conv_stem
#         self.bn1 = model.bn1
#         self.act1 = model.act1
#         self.block0 = model.blocks[0]
#         self.block1 = model.blocks[1]
#         self.block2 = model.blocks[2]
#         self.block3 = model.blocks[3:5]
#         self.block4 = model.blocks[5]
#
#         self.fpn_layer4 = FPNLayer(channels[0], channels[1])
#         self.fpn_layer3 = FPNLayer(channels[1], channels[2])
#         self.fpn_layer2 = FPNLayer(channels[2], channels[3])
#         self.fpn_layer1 = FPNLayer(channels[3], channels[4])
#
#         self.out_conv1 = BasicConv2d(channels[4], channels[3],
#                                     kernel_size=3, padding=1, padding_mode="zeros",
#                                     norm_layer=nn.BatchNorm2d)
#         self.out_conv = BasicConv2d(channels[3], channels[3],
#                                     kernel_size=3, padding=1, padding_mode="zeros",
#                                     norm_layer=nn.BatchNorm2d)
#
#         self.output_channels = channels[::-1]
#
#     def forward(self, images):
#         c1 = self.act1(self.bn1(self.conv_stem(images)))  # [bz, 32, H/2, W/2]
#         c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
#         c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
#         c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
#         c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
#         c5 = self.block4(c4)  # [bz, 160, H/32, W/32]
#
#         p4 = self.fpn_layer4(c5, c4)  # [bz, 96, H/16, W/16]
#         p3 = self.fpn_layer3(p4, c3)  # [bz, 32, H/8, W/8]
#         p2 = self.fpn_layer2(p3, c2)  # [bz, 24, H/4, W/4]
#         p1 = self.fpn_layer1(p2, c1)  # [bz, 16, H/2, W/2]
#         p1 = self.out_conv1(p1)
#         p1 = self.out_conv(p1)
#
#         return [p1, p2, p3, p4, c5]