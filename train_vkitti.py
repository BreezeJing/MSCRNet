import argparse
import os
import random
import csv
import copy
import time
import math
# from dataloader import my_load as ls
# from dataloader import MyDataLoad as DA
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging

from dataloader import vkitti2 as ls
from dataloader import KITTILoader as DA
from models import *

def setup_logging(save_dir):
    logging.basicConfig(
        filename=os.path.join(save_dir, 'training.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def save_metrics(save_dir, epoch, train_loss, test_loss, epe_loss, c3, max_acc, max_epo):
    file_exists = os.path.isfile(os.path.join(save_dir, 'metrics.csv'))
    with open(os.path.join(save_dir, 'metrics.csv'), 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'test_c1', 'epe_loss', 'test_c3', 'max_acc', 'max_epo']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_c1': test_loss,
            'epe_loss': epe_loss,
            'test_c3': c3,
            'max_acc': max_acc,
            'max_epo': max_epo
        })

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 50:
        lr = 0.0001
    elif epoch <= 200:
        lr = 0.0001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info(f"Adjusted learning rate to {lr}")

def train_one_epoch(model, device, dataloader, optimizer, epoch, model_name, lsp_channel, lsp_mode, affinity_settings, udc):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f'Epoch [{epoch}] Training')
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(loop):
        imgL = imgL_crop.to(device, dtype=torch.float)
        imgR = imgR_crop.to(device, dtype=torch.float)
        disp_true = disp_crop_L.to(device, dtype=torch.float)

        mask = (disp_true < 192).detach()

        optimizer.zero_grad()

        if model_name == 'stackhourglass':
            rgb_depth, disp_finetune, predr = model(imgL, imgR)
            rgb_depth = torch.squeeze(rgb_depth, 1)
            disp_finetune = torch.squeeze(disp_finetune, 1)
            predr = torch.squeeze(predr, 1)
            if epoch <= 1:
                a,b,c=1.2 ,1.5 ,1.3
            elif epoch <= 2:
                a,b,c=0.7, 1.3, 2
            elif epoch <= 15:
                a, b, c = 0.5, 0.7, 3.6
            else:
                a, b, c = 0.5, 0.7,  3.6
            loss = (a * F.smooth_l1_loss(rgb_depth[mask], disp_true[mask], reduction='mean') +
                    b * F.smooth_l1_loss(disp_finetune[mask], disp_true[mask], reduction='mean') +
                    c * F.smooth_l1_loss(predr[mask], disp_true[mask], reduction='mean') )

            if torch.isnan(loss):
                logging.error("Loss is NaN. Saving problematic disparity map.")
                disp_np = disp_crop_L.cpu().numpy().reshape(imgL_crop.size(2), imgL_crop.size(3))
                cv2.imwrite("nan_dis.png", disp_np)
                raise ValueError("Loss is NaN")

        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}")
    return avg_loss

def test_one_epoch(model, device, dataloader, epoch, model_name):
    model.eval()
    total_test_loss = 0.0
    total_epe_loss = 0.0
    total_c3 = 0.0
    with torch.no_grad():
        loop = tqdm(dataloader, desc=f'Epoch [{epoch}] Testing')
        for batch_idx, (imgL, imgR, disp_L) in enumerate(loop):
            imgL = imgL.to(device, dtype=torch.float)
            imgR = imgR.to(device, dtype=torch.float)
            disp_true = disp_L.to(device, dtype=torch.float)

            if model_name == 'stackhourglass':
                _, _, predr = model(imgL, imgR)
                predr = torch.squeeze(predr, 1)
            else:
                raise NotImplementedError(f"Model {model_name} not implemented.")

            prer = predr.cpu()
            true_disp = disp_true.cpu().numpy()
            pred_disp = prer.numpy()

            mask = true_disp < 192
            pred_disp, true_disp = pred_disp[mask], true_disp[mask]
            abs_diff = np.abs(true_disp - pred_disp)
            correct3 = np.mean((abs_diff > 3) & (abs_diff / np.abs(true_disp) > 0.05))
            correct1 = np.mean(abs_diff > 1)
            # abs_diff[~mask] = 0
            total_pixels = mask.sum()
            c1 = correct1*100 if total_pixels > 0 else 0.0
            c3 = correct3*100 if total_pixels > 0 else 0.0
            epe_loss = np.mean(abs_diff)

            # correct1 = (abs_diff > 1).sum()
            # correct3 = ((abs_diff > 3) | (abs_diff > true_disp * 0.05)).sum()



            # epe_loss = np.mean(abs_diff[mask]) if total_pixels > 0 else 0.0
            # c1 = correct1 / total_pixels if total_pixels > 0 else 0.0
            # c3 = correct3 / total_pixels if total_pixels > 0 else 0.0

            total_test_loss += c1
            total_epe_loss += epe_loss
            total_c3 += c3

            loop.set_postfix(epe_loss=epe_loss)

    avg_c1 = (total_test_loss / len(dataloader))
    avg_epe = total_epe_loss / len(dataloader)
    avg_c3 = (total_c3 / len(dataloader))

    logging.info(f"Epoch [{epoch}] Test 1-px Error: {avg_c1:.2f}%, EPE: {avg_epe:.4f}, 3-px Error: {avg_c3:.2f}%")
    return avg_c1, avg_epe, avg_c3

def main():
    parser = argparse.ArgumentParser(description="优化后的深度学习训练脚本")
    parser.add_argument('--maxdisp', type=int, default=192, help='Maximum disparity')
    parser.add_argument('--model_name', type=str, default='stackhourglass', help='Model name')
    parser.add_argument('--datapath', type=str, default=r"Z:\vkitti2/", help='Path to training data')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--loadmodel', type=str, default=r'G:\mobelnetV2\060.tar', help='Path to pretrained model')
    parser.add_argument('--savemodel', type=str, default='./241216vkitti/', help='Path to save models')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    # 设置保存目录
    os.makedirs(args.savemodel, exist_ok=True)
    setup_logging(args.savemodel)

    # 数据集加载
    all_left_img, all_right_img, all_dis, test_left_img, test_right_img, test_dis = ls.dataloader(args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_dis, True),
        batch_size=1, shuffle=True, num_workers=2, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_dis, False),
        batch_size=2, shuffle=False, num_workers=1, drop_last=False)

    # 模型初始化
    affinity_settings = {
        'win_w': 3,
        'win_h': 3,
        'dilation': [1, 2, 4, 8]
    }
    udc = True
    if args.model_name == 'stackhourglass':
        model = anet(
            maxdisp=args.maxdisp,
            use_concat_volume=True,
            struct_fea_c=4,
            fuse_mode='separate',
            affinity_settings=affinity_settings,
            udc=udc
        )
    elif args.model_name == 'basic':
        model = basic(args.maxdisp)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    model = nn.DataParallel(model)
    model.to(device)
    logging.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # 加载预训练模型
    if args.loadmodel and os.path.isfile(args.loadmodel):
        logging.info(f"Loading pretrained model from {args.loadmodel}")
        checkpoint = torch.load(args.loadmodel, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.warning(f"No pretrained model found at {args.loadmodel}")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    # 初始化变量
    min_px3 = float('inf')
    min_loss = float('inf')
    min_acc = 100.0
    max_epo = 0
    start_full_time = time.time()

    # 保存训练指标
    metrics = []

    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}/{args.epochs}")

        adjust_learning_rate(optimizer, epoch)

        # 训练
        train_loss = train_one_epoch(
            model, device, TrainImgLoader, optimizer, epoch,
            args.model_name, lsp_channel=4, lsp_mode='separate',
            affinity_settings=affinity_settings, udc=udc
        )

        # 测试
        test_c1, test_epe, test_c3 = test_one_epoch(
            model, device, TestImgLoader, epoch, args.model_name
        )

        # 更新最大准确率
        if test_c1 < min_acc:
            min_acc = test_c1
            max_epo = epoch

        logging.info(f"Current max accuracy: {min_acc:.2f}% at epoch {max_epo}")

        # 保存模型和指标
        if test_c3 < min_px3:
            min_px3 = test_c3
            # savefilename = os.path.join(args.savemodel, f'best_3px_epoch{epoch}.tar')
            savefilename = os.path.join(args.savemodel, f'best_3px.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'test_c1': test_c1,
                'test_epe': test_epe,
                'test_c3': test_c3
            }, savefilename)
            logging.info(f"Saved new best 3-px model at epoch {epoch} with 3-px error {test_c3:.2f}%")

        if train_loss < min_loss:
            min_loss = train_loss
            # savefilename = os.path.join(args.savemodel, f'best_loss_epoch{epoch}.tar')
            savefilename = os.path.join(args.savemodel, f'best_loss.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'test_c1': test_c1,
                'test_epe': test_epe,
                'test_c3': test_c3
            }, savefilename)
            logging.info(f"Saved new best loss model at epoch {epoch} with loss {train_loss:.4f}")

        # 记录指标
        save_metrics(args.savemodel, epoch, train_loss, test_c1, test_epe, test_c3, min_acc, max_epo)

    total_time = time.time() - start_full_time
    logging.info(f"Training completed in {total_time / 3600:.2f} hours")
    logging.info(f"Maximum accuracy {min_acc:.2f}% achieved at epoch {max_epo}")

if __name__ == '__main__':
    main()
