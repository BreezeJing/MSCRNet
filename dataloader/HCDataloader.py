import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    # ========== 路径定义 ==========
    train_left = os.path.join(filepath, 'train', 'LeftImg')
    train_right = os.path.join(filepath, 'train', 'RightImg')
    train_disp = os.path.join(filepath, 'train', 'Disp')

    val_left = os.path.join(filepath, 'val', 'LeftImg')
    val_right = os.path.join(filepath, 'val', 'RightImg')
    val_disp = os.path.join(filepath, 'val', 'Disp')

    # ========== 数据加载 ==========
    train_images = [img for img in os.listdir(train_left) if is_image_file(img)]
    val_images = [img for img in os.listdir(val_left) if is_image_file(img)]

    # ========== 训练集过滤 ==========
    left_train, right_train, disp_train_L = [], [], []
    for img in train_images:
        left_path = os.path.join(train_left, img)
        right_path = os.path.join(train_right, img)
        # 修正视差图路径：强制使用.png扩展名
        base_name = os.path.splitext(img)[0]
        disp_name = f"{base_name}.png"  # 视差图均为PNG
        disp_path = os.path.join(train_disp, disp_name)

        if all([os.path.isfile(right_path), os.path.isfile(disp_path)]):
            left_train.append(left_path)
            right_train.append(right_path)
            disp_train_L.append(disp_path)

    # ========== 验证集过滤 ==========
    left_val, right_val, disp_val_L = [], [], []
    for img in val_images:
        left_path = os.path.join(val_left, img)
        right_path = os.path.join(val_right, img)
        base_name = os.path.splitext(img)[0]
        disp_name = f"{base_name}.png"
        disp_path = os.path.join(val_disp, disp_name)

        if all([os.path.isfile(right_path), os.path.isfile(disp_path)]):
            left_val.append(left_path)
            right_val.append(right_path)
            disp_val_L.append(disp_path)

    # ========== 加载统计 ==========
    print("\n" + "=" * 40)
    print(f"训练集统计:".ljust(20) +
          f"原始样本 {len(train_images)} | 有效加载 {len(left_train)} | 过滤 {len(train_images) - len(left_train)}")
    print(f"验证集统计:".ljust(20) +
          f"原始样本 {len(val_images)} | 有效加载 {len(left_val)} | 过滤 {len(val_images) - len(left_val)}")
    print("=" * 40 + "\n")

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L