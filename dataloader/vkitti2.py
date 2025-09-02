import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.bmp'])

def dataloader(filepath, train_ratio=0.8):
    """
    加载并划分训练集和验证集。
    :param filepath: 数据集根目录
    :param train_ratio: 训练集比例
    :return: (train_left, train_right, train_depth), (val_left, val_right, val_depth)
    """
    views = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right','clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']  # 支持的视角
    all_left, all_right, all_depth = [], [], []

    scenes = sorted(os.listdir(os.path.join(filepath, 'RGB')))
    for scene in scenes:
        for view in views:
            left_path = os.path.join(filepath, 'RGB', scene, view, 'frames', 'rgb', 'Camera_0')
            right_path = os.path.join(filepath, 'RGB', scene, view, 'frames', 'rgb', 'Camera_1')
            depth_path = os.path.join(filepath, 'DISP', scene, view, 'frames', 'depth', 'Camera_0')


            if not (os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(depth_path)):
                continue

            left_images = sorted([os.path.join(left_path, f) for f in os.listdir(left_path) if is_image_file(f)])
            right_images = sorted([os.path.join(right_path, f) for f in os.listdir(right_path) if is_image_file(f)])
            depth_images = sorted([os.path.join(depth_path, f) for f in os.listdir(depth_path) if is_image_file(f)])
            # print(left_images,right_images,depth_images)
            all_left.extend(left_images)
            all_right.extend(right_images)
            all_depth.extend(depth_images)

    total_samples = len(all_left)
    train_size = int(total_samples * train_ratio)

    train_left, val_left = all_left[:train_size], all_left[train_size:]
    train_right, val_right = all_right[:train_size], all_right[train_size:]
    train_depth, val_depth = all_depth[:train_size], all_depth[train_size:]

    return train_left, train_right, train_depth, val_left, val_right, val_depth