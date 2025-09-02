import argparse
import os
import csv
import time
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
# from torch.cuda.amp import autocast, GradScaler
from dataloader import NV_Datasets as ls
from dataloader import NV_Datasets_Loader as DA
from models import anet, basic

def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
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

def save_metrics(save_dir, epoch, train_loss, err1, epe, err3, best_err3, best_epoch):
    path = os.path.join(save_dir, 'metrics.csv')
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['epoch','train_loss','1px_err(%)','EPE','3px_err(%)','best_3px(%)','best_epoch'])
        writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{err1:.2f}",
            f"{epe:.4f}",
            f"{err3:.2f}",
            f"{best_err3:.2f}",
            best_epoch
        ])

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 5:
        lr = 1e-3
    elif epoch <= 15:
        lr = 1e-3
    else:
        lr = 1e-4
    for g in optimizer.param_groups:
        g['lr'] = lr
    logging.info(f"Learning rate set to {lr}")

def train_one_epoch(model, device, loader, optimizer, epoch, model_name,
                    use_amp=False, max_batches=None):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    total_loss = 0.0
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(enumerate(loader), total=num_batches, desc=f"Epoch {epoch} Train")
    for idx, (imgL, imgR, dispL) in pbar:
        if max_batches is not None and idx >= max_batches:
            break

        imgL = imgL.to(device)
        imgR = imgR.to(device)
        disp_true = dispL.to(device)
        mask = disp_true < 192

        optimizer.zero_grad()
        with autocast('cuda', enabled=use_amp):
            if model_name == 'stackhourglass':
                rd, df, pr = model(imgL, imgR)
                rd = rd.squeeze(1)
                df = df.squeeze(1)
                pr = pr.squeeze(1)

                if epoch <= 2:
                    a, b, c = 2, 1.3, 0.7
                elif epoch <= 5:
                    a, b, c = 0.7, 2, 1.3
                elif epoch <= 12:
                    a, b, c = 0.7, 1.3, 2
                else:
                    a, b, c = 1.3, 1.3, 1.5

                loss = (
                    a * F.smooth_l1_loss(rd[mask], disp_true[mask]) +
                    b * F.smooth_l1_loss(df[mask], disp_true[mask]) +
                    c * F.smooth_l1_loss(pr[mask], disp_true[mask])
                )
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
        pbar.set_postfix(train_loss=f"{loss.item():.4f}")

    avg_loss = total_loss / (idx + 1)
    logging.info(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")
    return avg_loss

def test_one_epoch(model, device, loader, epoch, model_name, max_batches=None, maxdisp=192):
    model.eval()
    total_valid = 0
    total_abs = 0.0
    total_err1 = 0
    total_err3 = 0

    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(enumerate(loader), total=num_batches, desc=f"Epoch {epoch} Test")
    with torch.no_grad():
        for idx, (imgL, imgR, dispL) in pbar:
            if max_batches is not None and idx >= max_batches:
                break

            imgL = imgL.to(device)
            imgR = imgR.to(device)
            disp_true = dispL.to(device)

            if model_name == 'stackhourglass':
                _, _, pr = model(imgL, imgR)
                pr = pr.squeeze(1)
            else:
                raise NotImplementedError(f"Model {model_name} not implemented.")

            # valid mask
            valid_mask = disp_true < maxdisp
            n_valid = valid_mask.sum().item()
            if n_valid == 0:
                continue

            # absolute error
            diff = (pr - disp_true).abs()
            total_abs += diff[valid_mask].sum().item()
            total_err1 += (diff[valid_mask] > 1).sum().item()

            # relative for 3px
            true_vals = disp_true[valid_mask]
            diff_vals = diff[valid_mask]
            nonzero = true_vals != 0
            rel = torch.zeros_like(diff_vals)
            rel[nonzero] = diff_vals[nonzero] / true_vals[nonzero]
            cond3 = (diff_vals > 3) & (rel > 0.05)
            total_err3 += cond3.sum().item()

            total_valid += n_valid
            batch_epe = diff[valid_mask].mean().item()
            pbar.set_postfix(epe=f"{batch_epe:.4f}")

    if total_valid == 0:
        return 0.0, 0.0, 0.0

    avg_epe = total_abs / total_valid
    err1_pct = total_err1 / total_valid * 100
    err3_pct = total_err3 / total_valid * 100

    logging.info(
        f"Epoch {epoch} Test → 1px_err: {err1_pct:.2f}%, "
        f"EPE: {avg_epe:.4f}, 3px_err: {err3_pct:.2f}%"
    )
    return err1_pct, avg_epe, err3_pct

def main():
    parser = argparse.ArgumentParser(description="多显卡训练脚本")
    parser.add_argument('--maxdisp',      type=int,   default=192)
    parser.add_argument('--model_name',   type=str,   default='stackhourglass')
    parser.add_argument('--datapath',     type=str,   default=r'Z:\NV_Datasets/')
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--loadmodel',    type=str,   default=r'E:\Code\Stereo\250403HCDatasets\best_3px.pth')
    parser.add_argument('--savemodel',    type=str,   default='./checkpoints/')
    parser.add_argument('--gpu_ids',      type=str,   default='0')
    parser.add_argument('--train_batches',type=int,   default=5000, help="≤0 表示不限制")
    parser.add_argument('--test_batches', type=int,   default=-1, help="≤0 表示不限制")
    parser.add_argument('--amp',          default=False, help="启用混合精度")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_logging(args.savemodel)

    # model init
    affinity = {'win_w':3, 'win_h':3, 'dilation':[1,2,4,8]}
    if args.model_name == 'stackhourglass':
        model = anet(
            maxdisp=args.maxdisp,
            use_concat_volume=True,
            struct_fea_c=4,
            fuse_mode='separate',
            affinity_settings=affinity,
            udc=True
        )
    elif args.model_name == 'basic':
        model = basic(args.maxdisp)
    else:
        raise ValueError(f"Unknown model {args.model_name}")

    device_ids = [int(i) for i in args.gpu_ids.split(',')]
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    # load checkpoint if any
    if args.loadmodel and os.path.isfile(args.loadmodel):
        ck = torch.load(args.loadmodel, map_location=device, weights_only=False)
        sd = ck.get('state_dict', ck)
        sd = {k.replace('module.',''):v for k,v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logging.info(f"Loaded pretrained model from {args.loadmodel}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # data loaders
    left, right, disp, tleft, tright, tdisp = ls.dataloader(args.datapath)
    train_loader = torch.utils.data.DataLoader(
        DA.myImageFloder(left, right, disp, training=True),
        batch_size=2, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        DA.myImageFloder(tleft, tright, tdisp, training=False),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    tb = None if args.train_batches <= 0 else args.train_batches
    vb = None if args.test_batches  <= 0 else args.test_batches

    best = {'3px': float('inf'), 'loss': float('inf'), 'epoch': 0}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        train_loss = train_one_epoch(
            model, device, train_loader,
            optimizer, epoch, args.model_name,
            use_amp=args.amp, max_batches=tb
        )
        err1, epe, err3 = test_one_epoch(
            model, device, test_loader,
            epoch, args.model_name,
            max_batches=vb, maxdisp=args.maxdisp
        )

        # save best
        if err3 < best['3px']:
            best.update({'3px': err3, 'epoch': epoch})
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                '3px_err': err3
            }, os.path.join(args.savemodel, 'best_3px.pth'))

        if train_loss < best['loss']:
            best['loss'] = train_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'loss': train_loss
            }, os.path.join(args.savemodel, 'best_loss.pth'))

        save_metrics(
            args.savemodel, epoch,
            train_loss, err1, epe, err3,
            best['3px'], best['epoch']
        )

        torch.cuda.empty_cache()

    total_hrs = (time.time() - start_time) / 3600
    logging.info(f"Training completed in {total_hrs:.2f}h; best 3px_err={best['3px']:.2f}% @ epoch {best['epoch']}")

if __name__ == '__main__':
    main()
