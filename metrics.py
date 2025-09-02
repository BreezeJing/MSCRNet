import os
import cv2
import numpy as np
import csv
from tqdm import tqdm


def compute_metrics(pred, gt):
    """与之前相同的指标计算函数"""
    valid_mask = gt > 0
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    num_valid = np.sum(valid_mask)

    if num_valid == 0:
        return {k: 0.0 for k in ['EPE', 'D1-ALL', 'D2-ALL', 'D3-ALL']}

    abs_diff = np.abs(pred_valid - gt_valid)
    epe = np.mean(abs_diff)

    thresholds = [
        ('D1-ALL', 3),
        ('D2-ALL', 2),
        ('D3-ALL', 1)
    ]

    metrics = {'EPE': epe}
    for metric_name, px_threshold in thresholds:
        threshold = np.maximum(px_threshold, 0.05 * gt_valid)
        error_rate = np.mean(abs_diff > threshold) * 100
        metrics[metric_name] = error_rate

    return metrics


def batch_evaluate(pred_dir, gt_dir, output_csv, pred_scale=1.0/256.0, gt_scale=1.0):
    """
    批量评估视差图
    :param pred_dir: 预测视差图文件夹路径
    :param gt_dir: 真值视差图文件夹路径
    :param output_csv: 结果保存路径
    :param pred_scale: 预测视差值缩放因子 (默认无缩放)
    :param gt_scale: 真值视差值缩放因子 (如KITTI需设为1/256)
    """
    # 获取文件列表
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith('.png')])
    gt_files = set(os.listdir(gt_dir))

    results = []

    # 遍历处理每个文件
    for pred_file in tqdm(pred_files, desc='Processing'):
        # 验证文件存在性
        if pred_file not in gt_files:
            print(f"Warning: 缺失对应的真值文件 {pred_file}")
            continue

        # 读取图像
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, pred_file)

        # 使用OpenCV读取PNG (保持原始位深)
        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * pred_scale
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * gt_scale

        # 计算指标
        metrics = compute_metrics(pred, gt)

        # 记录结果
        results.append({
            'filename': pred_file,
            **metrics
        })

    # 保存结果到CSV
    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'EPE', 'D1-ALL', 'D2-ALL', 'D3-ALL'])
            writer.writeheader()

            # 写入各文件结果
            avg_metrics = {k: 0.0 for k in ['EPE', 'D1-ALL', 'D2-ALL', 'D3-ALL']}
            for row in results:
                writer.writerow(row)
                for k in avg_metrics:
                    avg_metrics[k] += row[k]

            # 计算并写入平均值
            num_files = len(results)
            avg_row = {'filename': 'AVERAGE'}
            for k, v in avg_metrics.items():
                avg_row[k] = v / num_files if num_files > 0 else 0.0
            writer.writerow(avg_row)

        print(f"结果已保存至 {output_csv}")
    else:
        print("未找到有效文件对")


if __name__ == "__main__":
    # 使用示例 (按需修改参数)
    batch_evaluate(
        pred_dir=r'G:\data\VAL_NSA\kiwi\depth',  # 预测视差图文件夹
        gt_dir=r'G:\data\VAL_NSA\kiwi\disp',  # 真值视差图文件夹
        output_csv='./evaluation_results_Kiwi2012_gen_BEST.csv',
        pred_scale=1.0 / 256,
        gt_scale=1.0 / 256  # KITTI数据集需要此缩放
    )