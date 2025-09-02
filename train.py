from __future__ import print_function
import argparse
import os
import random

from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from utils import *
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
# import skimage
# import skimage.io
# import skimage.transform
import numpy as np
import time
import math
import copy
import matplotlib.pyplot as plt

from dataloader import load as ls
from dataloader import KITTILoader as DA
# from dataloader import listflowfile as lt
# from dataloader import SecenFlowLoader as DA

from models import *
#-------------------变量-------------------------
maxdisp = 192 #maxium disparity
model_name = 'stackhourglass' #select model
datapath ="./training/"
# datapath = './data/'
epochs = 1000 #number of epochs to train
# loadmodel = None

loadmodel="./premodel.tar"
savemodel = './'
cuda = True  #enables CUDA training
seed = 1 #random seed (default: 1)

loss_train=[]
loss_test=[]
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


torch.cuda.manual_seed(seed)  #设置CPU生成随机数的种子，方便下次复现实验结果。
#数据集划分
all_left_img, all_right_img, all_dis, test_left_img, test_right_img, test_dis = ls.dataloader(datapath)

# dataset (Dataset) – 加载数据的数据集。
# batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
# shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
# sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
# num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
# collate_fn (callable, optional) –
# pin_memory (bool, optional) –
# drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
# ###

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img, all_right_img, all_dis, True),
         batch_size=1, shuffle= True, num_workers=2, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_dis, False),
         batch_size= 2, shuffle= False, num_workers=1, drop_last=False)

if model_name == 'stackhourglass':
    model = anet(maxdisp=maxdisp,use_concat_volume=True,struct_fea_c=lsp_channel, fuse_mode=lsp_mode,
                 affinity_settings=affinity_settings, udc=udc)
    # model = anet(maxdisp)

elif model_name == 'basic':
    model = basic(maxdisp)
else:
    print('no model')

if cuda:
    model = nn.DataParallel(model)
    model.cuda()

if loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters（模型参数的数量）: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
# params(iterable)：可用于迭代优化的参数或者定义参数组的dicts。
# lr (float, optional) ：学习率(默认: 1e-3)
# betas (Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))
# eps (float, optional)：为了提高数值稳定性而添加到分母的一个项(默认: 1e-8)
# weight_decay (float, optional)：权重衰减(如L2惩罚)(默认: 0)

def train(imgL, imgR, disp_L,x):
    left=imgL

    # training

    #model = 'stackhourglass'
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))
    # print(imgL.shape, imgR.shape, disp_L.shape)

    if cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    # print(imgL.size(),imgR.size(),disp_true.size())

    # ---------
    mask = (disp_true > 0)
    # print(mask.shape)
    mask.detach_()
    # ----
    optimizer.zero_grad()

    if model_name == 'stackhourglass':
        rgb_depth,disp_finetune,predr = model(imgL, imgR)

        disp_gt=disp_true
        rgb_depth = torch.squeeze(rgb_depth, 1)
        disp_finetune=torch.squeeze(disp_finetune, 1)
        predr = torch.squeeze(predr, 1)
        disp_ests = predr


        # print(output1)

        # print(output1.shape)
        # rgb_conf = torch.squeeze(rgb_conf, 1)

        # # print(output2.shape)
        # output3 = torch.squeeze(output3, 1)


        loss = 0.7 * F.smooth_l1_loss(rgb_depth[mask], disp_true[mask], reduction='mean') + (1) * F.smooth_l1_loss(
            disp_finetune[mask], disp_true[mask], reduction='mean') + 1.3 * F.smooth_l1_loss(
            predr[mask], disp_true[mask], reduction='mean')
        # scalar_outputs = {"loss": loss}
        # with torch.no_grad():
        #     # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
        #     scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        #     scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        #     scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
        #     scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
        #     scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]


        # loss=F.smooth_l1_loss(
        #  predr[mask], disp_true[mask], reduction='mean')
        if np.isnan(loss.cpu().detach().numpy())==True:
            print(disp_L)
            cv2.imwrite("nan_dis.png",disp_L.cpu().detach().numpy().reshape(int(left.size()[2]),int(left.size()[3])))
            input()







    # print(loss)

    loss.backward()
    optimizer.step()

    return loss.data


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    mask= (disp_true > 0)
    if cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        _, disp_finetune, prer = model(imgL, imgR)

    # pred_disp = disp_finetune.data.cpu()
    #
    # pred_disp = torch.squeeze(pred_disp,1)
    prer = prer.data.cpu()

    prer = torch.squeeze(prer, 1)
    # pred_disp = F.interpolate(pred_disp, [left.size()[2] / 4, left.size()[3] / 4], mode='trilinear', align_corners=True)


    # computing 3-px error#
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp > 0)
    if len(index[0])==0:
        print(index)
    # disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:] ,index[1][:], index[2][:]])
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - prer[index[0][:], index[1][:], index[2][:]])
    correct1 = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)
    correct3 = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
                disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
            index[0][:], index[1][:], index[2][:]] * 0.05)
    if len(disp_true[mask]) == 0:
        loss_epe = 0
    else:
        loss_epe = torch.mean(torch.abs(prer[mask] - true_disp[mask]))
    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct1)) / float(len(index[0]))),loss_epe,1 - (float(torch.sum(correct3)) / float(len(index[0])))


# def adjust_learning_rate(optimizer, epoch):
#     lr = 0.001
#     print(lr)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def adjust_learning_rate(optimizer, epoch):
    if epoch <= 20:
        lr = 0.001
    elif epoch <= 200:
        lr = 0.0001
    else:
        lr = 0.00001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    x=0
    min_px3=1000
    min_l=1000
    max_acc=0
    max_epo=0
    start_full_time = time.time()
    for epoch in range(1, epochs+1):
        total_train_loss = 0
        total_test_loss = 0
        total_epe_loss=0
        total_c3=0
        print('This is %d-th epoch' %(epoch))
        adjust_learning_rate(optimizer,epoch)
        loop=tqdm(TrainImgLoader)
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(loop):


            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L, x)

            # print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            l=total_train_loss/len(TrainImgLoader)
            loop.set_description(f'Epoch [{epoch+1/epochs}]')
            loop.set_postfix(loss=loss)
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        # loss_train.append((total_train_loss/len(TrainImgLoader)).cpu().detach().numpy().reshape(-1))



    #------------- TEST ------------------------------------------------------------
        # 第一列最后一块
        looptest = tqdm(TestImgLoader)
        for batch_idx, (imgL, imgR, disp_L) in enumerate(looptest):
                test_loss,epe_loss,c3 = test(imgL, imgR, disp_L)
                # print('Iter %d 3-px error in val = %.6f' % (batch_idx, test_loss * 100))
                total_test_loss += test_loss
                total_epe_loss += epe_loss
                total_c3 += c3
                looptest.set_description(f'Epoch [{epoch + 1 / epochs}]')
                looptest.set_postfix(loss=epe_loss)


        print('epoch %d total 1-px error in val = %.6f' %(epoch, total_test_loss/len(TestImgLoader)*100)+"epe=%.6f"%(total_epe_loss/len(TestImgLoader)))
        print("3px:",total_c3/len(TestImgLoader)*100)
        # loss_test.append(((total_test_loss/len(TrainImgLoader))*100).cpu().detach().numpy().reshape(-1))
        # fig, ax = plt.subplots(figsize=(12, 7))

        # 调用Axes对象的绘图接口，映射数据
        # ax.plot(x, loss_train)
        # ax.plot(x, loss_test)  # 多次调用将继续添加数据到图表
        # plt.savefig("loss")
        px3=total_test_loss/len(TestImgLoader)*100
        # if px3<60:
        #     x=1.0
        # elif px3<40:
        #     x=0.5
        # elif px3<20:
        #     x=0
        if total_test_loss/len(TestImgLoader)*100 > max_acc:
            max_acc = total_test_loss/len(TestImgLoader)*100
            max_epo = epoch
        print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))
        print("bili",x)


	#----------------------------------------------------------------------------------
	#SAVE test information
        if px3<min_px3:
            min_px3=px3

            savefilename = savemodel+'1_2best'+"_3px"+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                'test_loss': total_test_loss / len(TestImgLoader) * 100,
            }, savefilename)
        if l < min_l:
            min_l = l

            savefilename = savemodel + '1_2best'+ "_loss"+ '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                'test_loss': total_test_loss / len(TestImgLoader) * 100,
            }, savefilename)




        print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)

if __name__ == '__main__':
    main()