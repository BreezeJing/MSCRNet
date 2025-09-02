from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=r'G:\mobelnet\241107_kinect_kiwi_AQY\best_loss.tar',
                    help='loading model')
parser.add_argument('--leftimg', default=r'C:\Users\jingx\Desktop\left.png',
                    help='load model')
parser.add_argument('--rightimg', default=r'C:\Users\jingx\Desktop\right.png',
                    help='load model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
lsp_channel=4
no_udc=False
lsp_mode='separate'
affinity_settings = {}
lsp_width=3
lsp_dilation=[1, 2, 4, 8]
affinity_settings['win_w'] = lsp_width
affinity_settings['win_h'] = lsp_width
affinity_settings['dilation'] = lsp_dilation
udc = not no_udc
maxdisp=192

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = anet(maxdisp=maxdisp, use_concat_volume=True, struct_fea_c=lsp_channel, fuse_mode=lsp_mode,
                 affinity_settings=affinity_settings, udc=udc)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')


model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        # dis = dis.cuda()
    imgL = torch.unsqueeze(imgL, dim=0)
    imgR = torch.unsqueeze(imgR, dim=0)
    # dis = torch.unsqueeze(dis, dim=0)

    with torch.no_grad():
        _,_,disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])


    imgL_o = Image.open(args.leftimg).convert('RGB')
    imgR_o = Image.open(args.rightimg).convert('RGB')
    # dis_s=Image.open("./data/diss/000191_10.png")
    # w, h = imgL_o.size
    # imgL_o = imgL_o.crop((0, 0, 1024, 384))
    # imgR_o = imgR_o.crop((0, 0, 1024, 384))
    # imgL_o = imgL_o.resize((1280, 720))
    # imgR_o = imgR_o.resize((1280, 720))
    # imgL_o.save('cam1.png')
    # imgR_o.save('cam2.png')
    # imgL_o = imgL_o.crop((1000, 2000, 1000+1024, 2000+512))
    # imgR_o = imgR_o.crop((1000, 2000, 1000+1024, 2000+512))
    # imgL_o.save('resizel.png')
    # imgR_o.save('resizer.png')
    # imgL_o = imgL_o.crop((w - 1024, h - 256, w, h))
    # imgR_o = imgR_o.crop((w - 1024, h - 256, w, h))
    # dis_s = dis_s.crop((int(w - 1024), int(h - 256), int(w), int(h)))
    # dis_s = np.ascontiguousarray(dis_s, dtype=np.float32) / 256
    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)
    # dis= Variable(torch.FloatTensor(dis_s))
    # print(imgL.size(),imgR.size(),dis.size())





    start_time = time.time()
    pred_disp = test(imgL, imgR)
    print('time = %.2f' % (time.time() - start_time))
    # pred_disp=cv2.resize(pred_disp*1.725,(2208,1242))
    #
    # depth=(1060.83413 * 120)/(pred_disp)
    depth = (2976 * 19) / (pred_disp)
    depth_img = depth.astype('uint16')
    depth_img = Image.fromarray(depth_img)
    depth_img.save('depth_img.png')
    img = (pred_disp * 256).astype('uint16')
    img = Image.fromarray(img)
    # img = (pred_disp).astype('uint8')
    # img = Image.fromarray(img)
    img.save('Test_disparity.png')
    disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
        pred_disp, alpha=256 / 192), cv2.COLORMAP_JET)
    depth_color=cv2.applyColorMap(cv2.convertScaleAbs(
        depth, alpha=256 / 192), cv2.COLORMAP_JET)
    cv2.imwrite('disparity_leftRGB.bmp', disparity_left_color)
    cv2.imwrite('depth_RGB.bmp', depth_color)


if __name__ == '__main__':
    for i in range(2):
        main()