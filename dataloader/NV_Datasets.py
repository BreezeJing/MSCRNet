import os
import pickle
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    """判断是否是图像文件"""
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)

def find_image_with_stem(directory, stem):
    """
    在 directory 中查找以 stem 为文件名前缀的图像文件，
    如果找到则返回完整路径，否则返回 None。
    """
    for ext in IMG_EXTENSIONS:
        path = os.path.join(directory, stem + ext)
        if os.path.isfile(path):
            return path
    return None

def dataloader(root_dir, train_ratio=0.99, cache_filename='dataloader_cache.pkl'):
    """
    扫描 root_dir 下所有子序列，搜集 (left, right, disparity) 三元组，
    并按照 train_ratio 将数据切分为训练集和验证集。
    首次调用时会遍历目录并将结果缓存到 root_dir/cache_filename；
    以后调用将直接加载缓存，加速启动。

    Args:
        root_dir (str): 数据集根目录
        train_ratio (float): 训练集比例，范围 (0,1)，默认 0.9
        cache_filename (str): 缓存文件名，存放于 root_dir 下

    Returns:
        left_train, right_train, disp_train, left_val, right_val, disp_val
    """
    cache_path = os.path.join(root_dir, cache_filename)
    if os.path.exists(cache_path):
        # 直接加载缓存
        with open(cache_path, 'rb') as f:
            left_list, right_list, disp_list = pickle.load(f)
    else:
        left_list, right_list, disp_list = [], [], []
        # 遍历所有序列号文件夹
        for seq in os.listdir(root_dir):
            seq_path = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            # 遍历每个子数据集
            for ds in os.listdir(seq_path):
                base = os.path.join(seq_path, ds, 'dataset', 'data')
                left_rgb_dir  = os.path.join(base, 'left',  'rgb')
                left_disp_dir = os.path.join(base, 'left',  'disparity')
                right_rgb_dir = os.path.join(base, 'right', 'rgb')

                if not (os.path.isdir(left_rgb_dir) and
                        os.path.isdir(left_disp_dir) and
                        os.path.isdir(right_rgb_dir)):
                    continue

                # 匹配同名文件
                for fname in sorted(os.listdir(left_rgb_dir)):
                    if not is_image_file(fname):
                        continue
                    stem = os.path.splitext(fname)[0]
                    left_img  = os.path.join(left_rgb_dir,  fname)
                    disp_img  = find_image_with_stem(left_disp_dir, stem)
                    right_img = find_image_with_stem(right_rgb_dir, stem)

                    if disp_img and right_img:
                        left_list.append(left_img)
                        right_list.append(right_img)
                        disp_list.append(disp_img)

        # 将扫描结果写入缓存
        with open(cache_path, 'wb') as f:
            pickle.dump((left_list, right_list, disp_list), f)

    # 按比例切分
    total = len(left_list)
    split = int(total * train_ratio)
    left_train  = left_list[:split]
    right_train = right_list[:split]
    disp_train  = disp_list[:split]
    left_val    = left_list[split:]
    right_val   = right_list[split:]
    disp_val    = disp_list[split:]

    return left_train, right_train, disp_train, left_val, right_val, disp_val
