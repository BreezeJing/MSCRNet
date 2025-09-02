import argparse
import os
import random
import csv
import copy
import time
import math
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import logging
from dataloader import HCDataloader as ls
from dataloader import KITTILoaderEnhance as DA
# from dataloader import listflowfile as ls
# from dataloader import SecenFlowLoader as DA
# from dataloader import listflowfile as ls
# from dataloader import SecenFlowLoaderEnhance as DA
from models import anet, basic  # 根据实际模型定义修改


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


def train_one_epoch(model, device, dataloader, optimizer, epoch, model_name):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f'Epoch [{epoch}] Training')

    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(loop):
        imgL = imgL_crop.to(device)
        imgR = imgR_crop.to(device)
        disp_true = disp_crop_L.to(device)
        mask = (disp_true < 192).detach()

        optimizer.zero_grad()

        if model_name == 'stackhourglass':
            rgb_depth, disp_finetune, predr = model(imgL, imgR)
            rgb_depth = torch.squeeze(rgb_depth, 1)
            disp_finetune = torch.squeeze(disp_finetune, 1)
            predr = torch.squeeze(predr, 1)
            # a, b, c = 0.7, 1.3, 2
            if epoch <= 3:
                a, b, c = 2.0, 1.3, 0.7
            elif epoch <= 5:
                a, b, c = 0.7, 2.0, 1.3
            elif epoch == 50:
                a, b, c = 0.7, 1.3, 2.0
            elif epoch == 51:
                a, b, c = 0.7, 2.0, 1.3
            elif epoch == 100:
                a, b, c = 2.0, 1.3, 0.7
            elif epoch == 101:
                a, b, c = 0.7, 2.0, 1.3
            elif epoch <= 150:
                a, b, c = 0.5, 0.7, 3.6
            else:
                a, b, c = 0.0, 0.0, 5
            loss = (a * F.smooth_l1_loss(rgb_depth[mask], disp_true[mask], reduction='mean') +
                    b * F.smooth_l1_loss(disp_finetune[mask], disp_true[mask], reduction='mean') +
                    c * F.smooth_l1_loss(predr[mask], disp_true[mask], reduction='mean'))
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
    total_valid_pixels = 0
    total_abs_diff = 0.0
    total_correct1 = 0
    total_correct3 = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc=f'Epoch [{epoch}] Testing')
        for batch_idx, (imgL, imgR, disp_L) in enumerate(loop):
            imgL = imgL.to(device)
            imgR = imgR.to(device)
            disp_true = disp_L.to(device)

            if model_name == 'stackhourglass':
                _, _, predr = model(imgL, imgR)
                predr = torch.squeeze(predr, 1)
            else:
                raise NotImplementedError(f"Model {model_name} not implemented.")

            pred_disp = predr.cpu().numpy()
            true_disp = disp_true.cpu().numpy()
            mask = true_disp < 192
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

        logging.info(
            f"Epoch [{epoch}] Test 1-px Error: {avg_c1:.2f}%, EPE: {avg_epe:.4f}, 3-px Error: {avg_c3:.2f}%")
    return avg_c1, avg_epe, avg_c3


def main():
    parser = argparse.ArgumentParser(description="多显卡训练控制脚本")
    parser.add_argument('--maxdisp', type=int, default=192)
    parser.add_argument('--model_name', type=str, default='stackhourglass')
    parser.add_argument('--datapath', type=str, default=r"F:\HCDatasets/")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--loadmodel', type=str, default=r'F:\mobelnetV11\250305Scenceflow\best_loss.tar')
    parser.add_argument('--savemodel', type=str, default='./250312hc/')
    parser.add_argument('--gpu_ids', type=str, default='0', help='逗号分隔的显卡ID，如：0 或 0,1,2')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # 显卡控制逻辑
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device_ids = [int(id) for id in args.gpu_ids.split(',')]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Visible GPUs: {args.gpu_ids}")
    logging.info(f"Using device: {device}")

    # 初始化模型
    affinity_settings = {'win_w': 3, 'win_h': 3, 'dilation': [1, 2, 4, 8]}
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
        raise ValueError(f"Unknown model: {args.model_name}")

    # 多显卡

    # 加载预训练模型
    if args.loadmodel and os.path.isfile(args.loadmodel):
        logging.info(f"Loading model from {args.loadmodel}")
        checkpoint = torch.load(args.loadmodel, map_location=device)

        # 处理多显卡参数名称
        state_dict = checkpoint['state_dict']
        if isinstance(model, nn.DataParallel):
            # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            new_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in
                              state_dict.items()}
        else:
            # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # new_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v
            #                   for k, v in state_dict.items()}
            new_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in
                              state_dict.items()}

        model.load_state_dict(new_state_dict)
        logging.info("Model loaded successfully")

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

        logging.info(f"Using DataParallel on GPUs: {device_ids}")
    model = model.to(device)
    # 其余代码保持不变...
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.01)
    os.makedirs(args.savemodel, exist_ok=True)
    setup_logging(args.savemodel)

    # 数据加载
    all_left_img, all_right_img, all_dis, test_left_img, test_right_img, test_dis = ls.dataloader(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_dis, True),
        batch_size=6, shuffle=True, num_workers=0, pin_memory=True
    )
    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_dis, False),
        batch_size=2, shuffle=False, num_workers=0, pin_memory=True
    )

    # 训练循环
    min_px3 = float('inf')
    min_loss = float('inf')
    min_acc = 100.0
    max_epo = 0
    start_full_time = time.time()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train_loss = train_one_epoch(model, device, TrainImgLoader, optimizer, epoch, args.model_name)
        test_c1, test_epe, test_c3 = test_one_epoch(model, device, TestImgLoader, epoch, args.model_name)

        if test_c3 < min_acc:
            min_acc = test_c3
            max_epo = epoch

        # 模型保存逻辑...
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
        # ...（保持原样）


if __name__ == '__main__':
    main()