import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        #save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x *( torch.tanh(F.softplus(x)))
def convbnrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)
def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))
def convbn1(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)

def deconvbn(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn1(inplanes, planes, 3, stride, pad, dilation),
                                   Mish())

        self.conv2 = convbn1(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):#卷积
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):#卷积核为1
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)
class BasicBlockGeo(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            #norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def  warp(x, disp):
    """
    warp an image/tensor (imright) back to imleft, according to the disp

    x: [B, C, H, W] (imright)
    disp: [B, 1, H, W] disp

    """
    B, C, H, W = x.size()
    device = x.get_device()
    # mesh grid
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)#arange产生的是10-1=9 由1-9组成的1维度张量 ，类型int    repeat：复制张量H份    View：改变张量形状   view(1, -1)一维
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    xx = xx.float()
    yy = yy.float()
    # grid = torch.cat((xx, yy), 1).float()

#     if x.is_cuda:
#         xx = xx.float().cuda()
#         yy = yy.float().cuda()
    xx_warp = Variable(xx) - disp
    yy = Variable(yy)
    # print("xx_warp",xx.size(),yy.size())
#     xx_warp = xx - disp
    vgrid = torch.cat((xx_warp, yy), 1)
    # vgrid = Variable(grid) + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)#维度变换
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    mask = torch.ones(x.size(), device=device, requires_grad=True)
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask


def  warp_y(x, disp):
    """
    warp an image/tensor (imright) back to imleft, according to the disp

    x: [B, C, H, W] (imright)
    disp: [B, 1, H, W] disp

    """
    B, C, H, W = x.size()
    device = x.get_device()
    # mesh grid
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)#arange产生的是10-1=9 由1-9组成的1维度张量 ，类型int    repeat：复制张量H份    View：改变张量形状   view(1, -1)一维
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    xx = xx.float()
    yy = yy.float()
    # grid = torch.cat((xx, yy), 1).float()

#     if x.is_cuda:
#         xx = xx.float().cuda()
#         yy = yy.float().cuda()
    xx_warp = Variable(xx) - disp
    yy = Variable(yy)
    # print("xx_warp",xx.size(),yy.size())
#     xx_warp = xx - disp
    vgrid = torch.cat((xx_warp, yy), 1)
    # vgrid = Variable(grid) + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)#维度变换
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    mask = torch.ones(x.size(), device=device, requires_grad=True)
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask



def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    # print(cost.shape)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    # print(B, C, H, W)
    #[B,G,D,H,W]
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_corrleation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, 2 * maxdisp + 1, H, W])
    for i in range(-maxdisp, maxdisp+1):
        if i > 0:
            volume[:, :, i + maxdisp, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        elif i < 0:
            volume[:, :, i + maxdisp, :, :-i] = groupwise_correlation(refimg_fea[:, :, :, :-i],
                                                                     targetimg_fea[:, :, :, i:],
                                                                     num_groups)
        else:
            volume[:, :, i + maxdisp, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

class DisparityRegression(nn.Module):

    def __init__(self, maxdisp, win_size):
        super(DisparityRegression, self).__init__()
        self.max_disp = maxdisp
        self.win_size = win_size

    def forward(self, x):
        disp = torch.arange(0, self.max_disp).view(1, -1, 1, 1).float().to(x.device)

        if self.win_size > 0:
            max_d = torch.argmax(x, dim=1, keepdim=True)
            d_value = []
            prob_value = []
            for d in range(-self.win_size, self.win_size + 1):
                index = max_d + d
                index[index < 0] = 0
                index[index > x.shape[1] - 1] = x.shape[1] - 1
                d_value.append(index)

                prob = torch.gather(x, dim=1, index=index)
                prob_value.append(prob)

            part_x = torch.cat(prob_value, dim=1)
            part_x = part_x / (torch.sum(part_x, dim=1, keepdim=True) + 1e-8)
            part_d = torch.cat(d_value, dim=1).float()
            out = torch.sum(part_x * part_d, dim=1)

        else:
            out = torch.sum(x * disp, 1)

        return out