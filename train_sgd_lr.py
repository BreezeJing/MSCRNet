import argparse
import os
import random
import csv
import copy
import time
import math
import logging
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from dataloader import K2_K5_Loader as ls
# from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoadNew as DA
from models import anet  # 假设这是您的模型定义


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


def train_one_epoch(model, device, dataloader, optimizer, epoch, model_name, lsp_channel, lsp_mode, affinity_settings,
                    udc):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f'Epoch [{epoch}] Training')
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(loop):
        imgL = imgL_crop.to(device, dtype=torch.float)
        imgR = imgR_crop.to(device, dtype=torch.float)
        disp_true = disp_crop_L.to(device, dtype=torch.float)
        # mask = (disp_true > 0).detach()
        mask = ((disp_true > 0) & (disp_true < 192)).detach()  # 添加最大视差限制

        optimizer.zero_grad()

        if model_name == 'stackhourglass':
            rgb_depth, disp_finetune, predr = model(imgL, imgR)
            rgb_depth = torch.squeeze(rgb_depth, 1)
            disp_finetune = torch.squeeze(disp_finetune, 1)
            predr = torch.squeeze(predr, 1)
            a, b, c = 0.5, 1.3, 2
            # if epoch <= 15:
            #     a, b, c = 0.7, 1.3, 2
            # elif epoch <= 50:
            #     a, b, c = 0.5, 1.3, 2
            # elif epoch <= 50:
            #     a, b, c = 0.5, 1.3, 2
            # else:
            #     a, b, c = 0.5, 0.7, 2.8

            loss = (a * F.smooth_l1_loss(rgb_depth[mask], disp_true[mask], reduction='mean') +
                    b * F.smooth_l1_loss(disp_finetune[mask], disp_true[mask], reduction='mean') +
                    c * F.smooth_l1_loss(predr[mask], disp_true[mask], reduction='mean'))

            if torch.isnan(loss):
                logging.error("Loss is NaN. Saving problematic disparity map.")
                disp_np = disp_crop_L.cpu().numpy().reshape(imgL_crop.size(2), imgL_crop.size(3))
                cv2.imwrite("nan_dis.png", disp_np)
                raise ValueError("Loss is NaN")
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

        loss.backward()

        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
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

            pred_disp = predr.cpu().numpy()
            true_disp = disp_true.cpu().numpy()
            # mask = (true_disp > 0)
            mask = (true_disp > 0) & (true_disp < 192)  # 添加相同条件
            num_valid = mask.sum()
            if num_valid == 0:
                loop.set_postfix(epe_loss=0.0)
                continue

            pred_masked = pred_disp[mask]
            true_masked = true_disp[mask]
            abs_diff = np.abs(pred_masked - true_masked)

            total_abs_diff += abs_diff.sum()
            correct1 = (abs_diff > 1).sum()
            total_correct1 += correct1

            valid_rel = true_masked != 0
            rel_error = np.zeros_like(abs_diff)
            rel_error[valid_rel] = abs_diff[valid_rel] / true_masked[valid_rel]
            cond3 = (abs_diff > 3) & (rel_error > 0.05)
            correct3 = cond3.sum()
            total_correct3 += correct3
            total_valid_pixels += num_valid

            batch_epe = abs_diff.mean() if num_valid > 0 else 0.0
            loop.set_postfix(epe_loss=batch_epe)

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
    parser = argparse.ArgumentParser(description="使用SGD和余弦退火的完整训练脚本")
    parser.add_argument('--maxdisp', type=int, default=192, help='Maximum disparity')
    parser.add_argument('--model_name', type=str, default='stackhourglass', help='Model name')
    parser.add_argument('--datapath', type=str, default=r"F:\datasets\Kitti12_15/", help='Path to training data')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--loadmodel', type=str, default=r'G:\mobelnetV11v2\250318HCDatasets\best_loss.pth',
                        help='Path to pretrained model')
    parser.add_argument('--savemodel', type=str, default='./250317KITTI/', help='Path to save models')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
        batch_size=1,  # 增大batch_size以适应SGD
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_dis, False),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    # 模型初始化
    affinity_settings = {
        'win_w': 3,
        'win_h': 3,
        'dilation': [1, 2, 4, 8]
    }
    udc = True
    model = anet(
        maxdisp=args.maxdisp,
        use_concat_volume=True,
        struct_fea_c=4,
        fuse_mode='separate',
        affinity_settings=affinity_settings,
        udc=udc
    )

    model.to(device)
    logging.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    model = nn.DataParallel(model)
    # 加载预训练模型
    if args.loadmodel and os.path.isfile(args.loadmodel):
        logging.info(f"Loading pretrained model from {args.loadmodel}")
        checkpoint = torch.load(args.loadmodel, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict['state_dict'])
    else:
        logging.warning(f"No pretrained model found at {args.loadmodel}")

    # 优化器配置
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,  # 初始学习率
        momentum=0.9,  # 动量参数
        weight_decay=1e-5,  # 权重衰减
        nesterov=True
    )

    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,  # 完整周期数
        eta_min=1e-4  # 最小学习率
    )

    # 初始化训练指标
    min_px3 = float('inf')
    min_loss = float('inf')
    min_acc = 100.0
    max_epo = 0
    start_full_time = time.time()

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"\nEpoch {epoch}/{args.epochs} - Learning Rate: {current_lr:.2e}")

        # 训练阶段
        train_loss = train_one_epoch(
            model, device, TrainImgLoader, optimizer, epoch,
            args.model_name, lsp_channel=4, lsp_mode='separate',
            affinity_settings=affinity_settings, udc=udc
        )

        # 验证阶段
        test_c1, test_epe, test_c3 = test_one_epoch(
            model, device, TestImgLoader, epoch, args.model_name
        )

        # 更新学习率
        scheduler.step()

        # 更新最佳指标
        if test_c3 < min_acc:
            min_acc = test_c3
            max_epo = epoch

        # 保存最佳模型
        if test_c3 < min_px3:
            min_px3 = test_c3
            savefilename = os.path.join(args.savemodel, f'best_3px.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_c1': test_c1,
                'test_epe': test_epe,
                'test_c3': test_c3
            }, savefilename)
            logging.info(f"Saved new best 3-px model at epoch {epoch}")

        if train_loss < min_loss:
            min_loss = train_loss
            savefilename = os.path.join(args.savemodel, f'best_loss.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_c1': test_c1,
                'test_epe': test_epe,
                'test_c3': test_c3
            }, savefilename)
            logging.info(f"Saved new best loss model at epoch {epoch}")

        # 保存训练指标
        save_metrics(args.savemodel, epoch, train_loss, test_c1, test_epe, test_c3, min_acc, max_epo)

    # 最终输出
    total_time = time.time() - start_full_time
    logging.info(f"\nTraining completed in {total_time / 3600:.2f} hours")
    logging.info(f"Best 3-px Error: {min_acc:.2f}% achieved at epoch {max_epo}")


if __name__ == '__main__':
    main()