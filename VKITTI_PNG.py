import os
import shutil
import random
from tqdm import tqdm


def preprocess_and_split(source_dir, target_dir, train_ratio=0.8, seed=42):
    """
    在保存时直接划分训练集/验证集
    :param source_dir: 原始VKITTI2数据集路径
    :param target_dir: 新格式数据集保存路径
    :param train_ratio: 训练集比例
    :param seed: 随机种子保证可重复性
    """
    # 初始化随机种子
    random.seed(seed)

    # 场景和视角定义
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    views = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right',
             'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']

    # 收集所有有效样本路径
    samples = []
    for scene in tqdm(scenes, desc="Collecting samples"):
        for view in views:
            # 左图路径 (Camera_0)
            left_rgb_dir = os.path.join(source_dir, 'RGB', scene, view, 'frames', 'rgb', 'Camera_0')
            # 右图路径 (Camera_1)
            right_rgb_dir = os.path.join(source_dir, 'RGB', scene, view, 'frames', 'rgb', 'Camera_1')
            # 视差图路径 (Camera_0对应的深度)
            disp_dir = os.path.join(source_dir, 'DISP', scene, view, 'frames', 'depth', 'Camera_0')
            # print(left_rgb_dir,right_rgb_dir,disp_dir)
            # 跳过不完整的数据
            if not (os.path.exists(left_rgb_dir) and os.path.exists(right_rgb_dir) and os.path.exists(disp_dir)):
                continue

            # 获取所有左图文件名
            left_images = sorted([f for f in os.listdir(left_rgb_dir) if f.endswith(('.png', '.jpg'))])
            for img_name in left_images:
                left_src = os.path.join(left_rgb_dir, img_name)
                right_src = os.path.join(right_rgb_dir, img_name)
                # disp_src = os.path.join(disp_dir,img_name)  # 假设视差图后缀为.png
                disp_src = os.path.join(disp_dir, img_name.replace('rgb', 'depth').replace('.jpg', '.png'))  # 假设视差图后缀为.png
                # print(left_src,right_src,disp_src)
                # 验证三组文件存在
                if os.path.exists(right_src) and os.path.exists(disp_src):
                    # print('yes')
                    samples.append((left_src, right_src, disp_src))

    # 打乱样本顺序
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # 创建目标目录结构
    dirs_to_create = [
        os.path.join(target_dir, 'train', 'LeftImg'),
        os.path.join(target_dir, 'train', 'RightImg'),
        os.path.join(target_dir, 'train', 'Disp'),
        os.path.join(target_dir, 'val', 'LeftImg'),
        os.path.join(target_dir, 'val', 'RightImg'),
        os.path.join(target_dir, 'val', 'Disp')
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    # 全局计数器（保证文件名唯一）
    counter = 1

    # 处理训练集
    with tqdm(total=len(train_samples), desc="Copying TRAIN set") as pbar:
        for left_src, right_src, disp_src in train_samples:
            new_name = f"VK_{counter:08d}.png"  # 统一保存为PNG格式
            # 复制左图
            shutil.copy(left_src, os.path.join(target_dir, 'train', 'LeftImg', new_name))
            # 复制右图
            shutil.copy(right_src, os.path.join(target_dir, 'train', 'RightImg', new_name))
            # 复制视差图（确保转换为PNG）
            shutil.copy(disp_src, os.path.join(target_dir, 'train', 'Disp', new_name))
            counter += 1
            pbar.update(1)

    # 处理验证集
    with tqdm(total=len(val_samples), desc="Copying VAL set") as pbar:
        for left_src, right_src, disp_src in val_samples:
            new_name = f"VK_{counter:08d}.png"
            shutil.copy(left_src, os.path.join(target_dir, 'val', 'LeftImg', new_name))
            shutil.copy(right_src, os.path.join(target_dir, 'val', 'RightImg', new_name))
            shutil.copy(disp_src, os.path.join(target_dir, 'val', 'Disp', new_name))
            counter += 1
            pbar.update(1)

    print(f"处理完成! 总样本数: {len(samples)}")
    print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")



preprocess_and_split(
    source_dir=r'Y:\vkitti2/',
    target_dir=r'F:\HCDatasets/',
    train_ratio=0.8,
    seed=42
)

