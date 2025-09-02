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
# from dataloader import listflowfile as ls
# from dataloader import SecenFlowLoader as DA
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoadNew as DA
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
    if epoch <= 15:
        lr = 0.001
    elif epoch <= 150:
        lr = 0.0001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info(f"Adjusted learning rate to {lr}")


def train_one_epoch(model, device, dataloader, optimizer, epoch, model_name, lsp_channel, lsp_mode, affinity_settings,
                    udc):
    model.train()
    total_loss = 0.0
    valid_batches = 0  # 记录有效batch计数
    loop = tqdm(dataloader, desc=f'Epoch [{epoch}] Training')

    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(loop):
        imgL = imgL_crop.to(device, dtype=torch.float)
        imgR = imgR_crop.to(device, dtype=torch.float)
        disp_true = disp_crop_L.to(device, dtype=torch.float)

        # 数据有效性检查
        mask = (disp_true > 0).detach()
        if mask.sum() == 0:
            logging.warning(f"Batch {batch_idx} has no valid disparity. Skipping.")
            loop.set_postfix(skipped=True)
            continue

        optimizer.zero_grad()

        try:
            if model_name == 'stackhourglass':
                # 前向传播
                rgb_depth, disp_finetune, predr = model(imgL, imgR)
                rgb_depth = torch.squeeze(rgb_depth, 1)
                disp_finetune = torch.squeeze(disp_finetune, 1)
                predr = torch.squeeze(predr, 1)

                # 输出值检查
                if torch.isnan(rgb_depth).any() or torch.isnan(disp_finetune).any():
                    logging.warning(f"NaN detected in model outputs at batch {batch_idx}")
                    loop.set_postfix(nan_detected=True)
                    continue

                # 损失计算
                a, b, c = 0.7, 1.3, 2
                loss = (a * F.smooth_l1_loss(rgb_depth[mask], disp_true[mask], reduction='mean') +
                        b * F.smooth_l1_loss(disp_finetune[mask], disp_true[mask], reduction='mean') +
                        c * F.smooth_l1_loss(predr[mask], disp_true[mask], reduction='mean'))

                # 损失值检查
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss at batch {batch_idx}")
                    loop.set_postfix(invalid_loss=True)
                    continue

                # 反向传播和参数更新
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # 累计有效loss
                total_loss += loss.item()
                valid_batches += 1
                loop.set_postfix(loss=loss.item(), skipped=False)

            else:
                raise NotImplementedError(f"Model {model_name} not implemented.")

        except RuntimeError as e:
            if 'CUDA' in str(e):
                logging.error(f"CUDA error at batch {batch_idx}: {str(e)}")
            else:
                logging.error(f"Runtime error at batch {batch_idx}: {str(e)}")
            loop.set_postfix(error=True)
            torch.cuda.empty_cache()
            continue

    # 处理全NaN epoch的情况
    if valid_batches == 0:
        avg_loss = float('nan')
        logging.warning(f"Epoch {epoch} had no valid batches!")
    else:
        avg_loss = total_loss / valid_batches

    logging.info(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}")
    return avg_loss

def test_one_epoch(model, device, dataloader, epoch, model_name):
    model.eval()
    total_valid_pixels = 0
    total_abs_diff = 0.0
    total_correct1 = 0
    total_correct3 = 0
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

            # 转换为numpy数组
            pred_disp = predr.cpu().numpy()
            true_disp = disp_true.cpu().numpy()

            # 创建有效像素mask（基于真实视差）
            mask = ((true_disp > 0)&(true_disp < 192))
            num_valid = mask.sum()
            if num_valid == 0:
                # 没有有效像素，跳过该样本
                loop.set_postfix(epe_loss=0.0)
                continue
            # 应用mask到预测和真实视差
            pred_masked = pred_disp[mask]
            true_masked = true_disp[mask]
            abs_diff = np.abs(pred_masked - true_masked)
            # 累计绝对误差总和
            total_abs_diff += abs_diff.sum()
            # 计算correct1数目（误差>1像素）
            correct1 = (abs_diff > 1).sum()
            total_correct1 += correct1
            # 计算correct3数目（误差>3像素或相对误差>5%）
            valid_rel = true_masked != 0
            rel_error = np.zeros_like(abs_diff)
            rel_error[valid_rel] = abs_diff[valid_rel] / true_masked[valid_rel]
            cond3 = (abs_diff > 3) & (rel_error > 0.05)
            correct3 = cond3.sum()
            total_correct3 += correct3
            total_valid_pixels += num_valid
            # 更新进度条显示当前batch的EPE
            batch_epe = abs_diff.mean() if num_valid > 0 else 0.0
            loop.set_postfix(epe_loss=batch_epe)

        # 计算总指标
        if total_valid_pixels == 0:
            avg_c1 = 0.0
            avg_c3 = 0.0
            avg_epe = 0.0
        else:
            avg_epe = total_abs_diff / total_valid_pixels
            avg_c1 = (total_correct1 / total_valid_pixels) * 100
            avg_c3 = (total_correct3 / total_valid_pixels) * 100

        logging.info(f"Epoch [{epoch}] Test 1-px Error: {avg_c1:.2f}%, EPE: {avg_epe:.4f}, 3-px Error: {avg_c3:.2f}%")

    return avg_c1, avg_epe, avg_c3

def main():
    parser = argparse.ArgumentParser(description="优化后的深度学习训练脚本")
    parser.add_argument('--maxdisp', type=int, default=192, help='Maximum disparity')
    parser.add_argument('--model_name', type=str, default='stackhourglass', help='Model name')
    parser.add_argument('--datapath', type=str, default=r"G:\data\KIWI_LIDAR_DATASET/", help='Path to training data')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--loadmodel', type=str, default=r'G:\mobelnetV11v2\250318HCDatasets\best_loss.pth', help='Path to pretrained model')
    parser.add_argument('--savemodel', type=str, default='./250324KS/', help='Path to save models')
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
        batch_size=1, shuffle=False, num_workers=1, drop_last=False)

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


    model.to(device)
    logging.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # 加载预训练模型
    if args.loadmodel and os.path.isfile(args.loadmodel):
        logging.info(f"Loading pretrained model from {args.loadmodel}")
        checkpoint = torch.load(args.loadmodel, map_location=device)
        print("Checkpoint的键:", checkpoint['state_dict'].keys())
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        logging.warning(f"No pretrained model found at {args.loadmodel}")
    model = nn.DataParallel(model)
    print("Missing keys (需重新初始化的层):", missing_keys)
    print("Unexpected keys (旧模型多余的参数):", unexpected_keys)
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
        if test_c3 < min_acc:
            min_acc = test_c3
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
