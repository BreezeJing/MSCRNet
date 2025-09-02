import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import json

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def build_paths(root_dir, split):
    """构建指定数据分割的路径列表"""
    meta_path = os.path.join(root_dir, split, 'metadata.json')

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    left_dir = os.path.join(root_dir, split, 'left')
    right_dir = os.path.join(root_dir, split, 'right')
    disp_dir = os.path.join(root_dir, split, 'disparity')

    left_paths = []
    right_paths = []
    disp_paths = []

    for sample in metadata['samples']:
        base_name = sample['new_name']
        left = os.path.join(left_dir, base_name)
        right = os.path.join(right_dir, base_name)
        disp = os.path.join(disp_dir, base_name)

        if all(os.path.exists(p) for p in [left, right, disp]):
            left_paths.append(left)
            right_paths.append(right)
            disp_paths.append(disp)
        else:
            print(f"警告：缺失样本 {base_name}")

    return left_paths, right_paths, disp_paths


def dataloader(filepath):
    """新版数据加载器

    参数：
        filepath : 数据集根目录（包含train/和val/的目录）

    返回：
        (left_train, right_train, disp_train,
         left_val, right_val, disp_val)
    """
    # 构建训练集路径
    left_train, right_train, disp_train = build_paths(filepath, 'train')

    # 构建验证集路径
    left_val, right_val, disp_val = build_paths(filepath, 'val')

    # 完整性检查
    assert len(left_train) == len(right_train) == len(disp_train), "训练集数据不匹配"
    assert len(left_val) == len(right_val) == len(disp_val), "验证集数据不匹配"

    print(f"加载完成：训练集 {len(left_train)} 样本，验证集 {len(left_val)} 样本")

    return left_train, right_train, disp_train, left_val, right_val, disp_val