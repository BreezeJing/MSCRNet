import argparse
import os
import random
import csv
import time
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
# from dataloader import HCDataloader as ls
# from dataloader import KITTILoaderEnhance as DA
from dataloader import listflowfile as ls
from dataloader import SecenFlowLoaderEnhance as DA
from models import anet, basic
import numpy as np

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
    if epoch <= 2:
        lr = 0.01
    elif epoch <= 5:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info(f"Adjusted learning rate to {lr}")


def train_one_epoch(model, device, dataloader, optimizer, epoch, model_name, use_amp=False, max_batches=None):
    model.train()
    total_loss = 0.0
    scaler = GradScaler('cuda',enabled=use_amp)

    loop = tqdm(enumerate(dataloader), total=max_batches, desc=f'Epoch [{epoch}] Training')
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in loop:
        if max_batches and batch_idx >= max_batches:
            break

        imgL = imgL_crop.to(device)
        imgR = imgR_crop.to(device)
        disp_true = disp_crop_L.to(device)
        mask = (disp_true < 192).detach()

        optimizer.zero_grad()

        with autocast('cuda',enabled=use_amp):
            if model_name == 'stackhourglass':
                rgb_depth, disp_finetune, predr = model(imgL, imgR)
                rgb_depth = torch.squeeze(rgb_depth, 1)
                disp_finetune = torch.squeeze(disp_finetune, 1)
                predr = torch.squeeze(predr, 1)

                # 系数配置逻辑
                if epoch <= 2:
                    a, b, c = 2.0, 1.3, 0.7
                elif epoch <= 5:
                    a, b, c = 0.7, 2.0, 1.3
                elif epoch == 500:
                    a, b, c = 0.7, 1.3, 2.0
                elif epoch == 501:
                    a, b, c = 0.7, 2.0, 1.3
                elif epoch == 1000:
                    a, b, c = 2.0, 1.3, 0.7
                elif epoch == 1001:
                    a, b, c = 0.7, 2.0, 1.3
                elif epoch <= 1200:
                    a, b, c = 0.7, 1.3, 2.0
                else:
                    a, b, c = 0.0, 0.0, 5

                loss = (a * F.smooth_l1_loss(rgb_depth[mask], disp_true[mask], reduction='mean') +
                        b * F.smooth_l1_loss(disp_finetune[mask], disp_true[mask], reduction='mean') +
                        c * F.smooth_l1_loss(predr[mask], disp_true[mask], reduction='mean'))
            else:
                raise NotImplementedError(f"Model {model_name} not implemented.")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / (batch_idx + 1)
    logging.info(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}")
    return avg_loss


def test_one_epoch(model, device, dataloader, epoch, model_name, max_batches=None):
    model.eval()
    total_valid_pixels = 0
    total_abs_diff = 0.0
    total_correct1 = 0
    total_correct3 = 0

    with torch.no_grad():
        loop = tqdm(enumerate(dataloader), total=max_batches, desc=f'Epoch [{epoch}] Testing')
        for batch_idx, (imgL, imgR, disp_L) in loop:
            if max_batches and batch_idx >= max_batches:
                break

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
                loop.set_postfix(epe_loss=0.0)
                continue

            pred_masked = pred_disp[mask]
            true_masked = true_disp[mask]
            abs_diff = np.abs(pred_masked - true_masked)

            total_abs_diff += abs_diff.sum()
            total_correct1 += (abs_diff > 1).sum()

            valid_rel = true_masked != 0
            rel_error = np.zeros_like(abs_diff)
            rel_error[valid_rel] = abs_diff[valid_rel] / true_masked[valid_rel]
            cond3 = (abs_diff > 3) & (rel_error > 0.05)
            total_correct3 += cond3.sum()
            total_valid_pixels += num_valid

            batch_epe = abs_diff.mean() if num_valid > 0 else 0.0
            loop.set_postfix(epe_loss=batch_epe)

    if total_valid_pixels == 0:
        return 0.0, 0.0, 0.0

    avg_epe = total_abs_diff / total_valid_pixels
    avg_c1 = (total_correct1 / total_valid_pixels) * 100
    avg_c3 = (total_correct3 / total_valid_pixels) * 100

    logging.info(f"Epoch [{epoch}] Test 1-px Error: {avg_c1:.2f}%, EPE: {avg_epe:.4f}, 3-px Error: {avg_c3:.2f}%")
    return avg_c1, avg_epe, avg_c3


def main():
    parser = argparse.ArgumentParser(description="多显卡训练控制脚本")
    parser.add_argument('--maxdisp', type=int, default=192)
    parser.add_argument('--model_name', type=str, default='stackhourglass')
    parser.add_argument('--datapath', type=str,  default=r'E:\Data\stereo_matching\SceneFlow/')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--loadmodel', type=str, default=r'E:\Code\Stereo\mobelnetV11v2\250305Scenceflow\best_loss.tar')
    parser.add_argument('--savemodel', type=str, default='./250317SC/')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--train_batches', type=int, default=10000, help="每个训练epoch的最大批次")
    parser.add_argument('--test_batches', type=int, default=1500, help="每个测试epoch的最大批次")
    parser.add_argument('--amp', default=False, help="启用混合精度训练")
    args = parser.parse_args()

    # 初始化设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.savemodel, exist_ok=True)
    setup_logging(args.savemodel)

    # 模型初始化
    affinity_settings = {'win_w': 3, 'win_h': 3, 'dilation': [1, 2, 4, 8]}
    if args.model_name == 'stackhourglass':
        model = anet(
            maxdisp=args.maxdisp,
            use_concat_volume=True,
            struct_fea_c=4,
            fuse_mode='separate',
            affinity_settings=affinity_settings,
            udc=True
        )
    elif args.model_name == 'basic':
        model = basic(args.maxdisp)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    # 多GPU并行
    device_ids = [int(i) for i in args.gpu_ids.split(',')]
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    # 加载预训练权重
    if args.loadmodel and os.path.isfile(args.loadmodel):
        checkpoint = torch.load(args.loadmodel, map_location=device)
        state_dict = checkpoint['state_dict']
        # 统一处理多GPU参数前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded pretrained model from {args.loadmodel}")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    # 数据加载
    all_left, all_right, all_disp, test_left, test_right, test_disp = ls.dataloader(args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left, all_right, all_disp, training=True),
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left, test_right, test_disp, training=False),
        batch_size=2,
        shuffle=True,  # 测试集启用随机采样
        num_workers=2,
        pin_memory=True
    )

    # 训练循环
    best_metrics = {'c3': float('inf'), 'loss': float('inf'), 'epoch': 0}
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        # 训练阶段
        train_loss = train_one_epoch(
            model, device, TrainImgLoader, optimizer, epoch,
            args.model_name, args.amp, args.train_batches
        )
        # 测试阶段
        test_c1, test_epe, test_c3 = test_one_epoch(
            model, device, TestImgLoader, epoch,
            args.model_name, args.test_batches
        )

        # 保存模型逻辑修复
        current_best_c3 = best_metrics['c3']
        current_best_loss = best_metrics['loss']

        # 检查并更新最佳指标，同时保存模型
        if test_c3 < current_best_c3:
            best_metrics.update(c3=test_c3, epoch=epoch)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'metrics': (test_c1, test_epe, test_c3)
            }, os.path.join(args.savemodel, 'best_3px.pth'))

        if train_loss < current_best_loss:
            best_metrics.update(loss=train_loss)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'loss': train_loss
            }, os.path.join(args.savemodel, 'best_loss.pth'))

        # 记录指标
        save_metrics(
            args.savemodel, epoch, train_loss, test_c1,
            test_epe, test_c3, best_metrics['c3'], best_metrics['epoch']
        )

    # 最终报告
    total_time = time.time() - start_time
    logging.info(f"训练完成，用时 {total_time / 3600:.1f} 小时")
    logging.info(f"最佳3px误差: {best_metrics['c3']:.2f}% @ epoch {best_metrics['epoch']}")


if __name__ == '__main__':
    main()