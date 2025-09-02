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


def create_logs_directory(filepath):
    """创建日志目录和清单文件"""
    log_dir = os.path.join(filepath, 'data_loading_logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def write_list_to_file(file_path, items):
    """将列表写入文件"""
    with open(file_path, 'w') as f:
        for item in items:
            f.write(f"{item}\n")


def record_filtered_info(log_dir, dataset_type, filename, missing_files):
    """记录被过滤的文件信息"""
    log_file = os.path.join(log_dir, f'filtered_{dataset_type}.txt')
    with open(log_file, 'a') as f:
        f.write(f"{filename}: Missing {', '.join(missing_files)}\n")


def dataloader(filepath):
    # 创建日志目录
    log_dir = create_logs_directory(filepath)

    # ========== 路径定义 ==========
    train_left = os.path.join(filepath, 'train', 'LeftImg')
    train_right = os.path.join(filepath, 'train', 'RightImg')
    train_disp = os.path.join(filepath, 'train', 'Disp')

    val_left = os.path.join(filepath, 'val', 'LeftImg')
    val_right = os.path.join(filepath, 'val', 'RightImg')
    val_disp = os.path.join(filepath, 'val', 'Disp')

    # ========== 初始化记录列表 ==========
    loaded_records = {
        'train': {'left': [], 'right': [], 'disp': []},
        'val': {'left': [], 'right': [], 'disp': []}
    }
    filtered_records = {'train': [], 'val': []}

    # ========== 处理训练集 ==========
    raw_train_count = len(os.listdir(train_left))
    left_train, right_train, disp_train_L = [], [], []

    for img in [f for f in os.listdir(train_left) if is_image_file(f)]:
        missing = []
        left_path = os.path.join(train_left, img)
        right_path = os.path.join(train_right, img)
        disp_path = os.path.join(train_disp, img)

        # 检查文件是否存在
        right_exist = os.path.isfile(right_path)
        disp_exist = os.path.isfile(disp_path)

        if not right_exist:
            missing.append('right image')
        if not disp_exist:
            missing.append('disparity')

        if right_exist and disp_exist:
            left_train.append(left_path)
            right_train.append(right_path)
            disp_train_L.append(disp_path)
            # 记录已加载文件
            loaded_records['train']['left'].append(left_path)
            loaded_records['train']['right'].append(right_path)
            loaded_records['train']['disp'].append(disp_path)
        else:
            filtered_records['train'].append((img, missing))
            record_filtered_info(log_dir, 'train', img, missing)

    # ========== 处理验证集 ==========
    raw_val_count = len(os.listdir(val_left))
    left_val, right_val, disp_val_L = [], [], []

    for img in [f for f in os.listdir(val_left) if is_image_file(f)]:
        missing = []
        left_path = os.path.join(val_left, img)
        right_path = os.path.join(val_right, img)
        disp_path = os.path.join(val_disp, img)

        # 检查文件是否存在
        right_exist = os.path.isfile(right_path)
        disp_exist = os.path.isfile(disp_path)

        if not right_exist:
            missing.append('right image')
        if not disp_exist:
            missing.append('disparity')

        if right_exist and disp_exist:
            left_val.append(left_path)
            right_val.append(right_path)
            disp_val_L.append(disp_path)
            # 记录已加载文件
            loaded_records['val']['left'].append(left_path)
            loaded_records['val']['right'].append(right_path)
            loaded_records['val']['disp'].append(disp_path)
        else:
            filtered_records['val'].append((img, missing))
            record_filtered_info(log_dir, 'val', img, missing)

    # ========== 保存加载清单 ==========
    for dataset in ['train', 'val']:
        # 保存已加载文件路径
        write_list_to_file(
            os.path.join(log_dir, f'loaded_{dataset}_left.txt'),
            loaded_records[dataset]['left']
        )
        write_list_to_file(
            os.path.join(log_dir, f'loaded_{dataset}_right.txt'),
            loaded_records[dataset]['right']
        )
        write_list_to_file(
            os.path.join(log_dir, f'loaded_{dataset}_disp.txt'),
            loaded_records[dataset]['disp']
        )

    # ========== 打印统计信息 ==========
    print("\n" + "=" * 50)
    print(
        f"{'训练集统计:':<20} 原始样本 {raw_train_count:>5} | 有效加载 {len(left_train):>5} | 过滤 {len(filtered_records['train']):>5}")
    print(
        f"{'验证集统计:':<20} 原始样本 {raw_val_count:>5} | 有效加载 {len(left_val):>5} | 过滤 {len(filtered_records['val']):>5}")
    print("=" * 50)

    # 打印过滤文件示例
    print("\n过滤文件清单已保存至目录:", log_dir)
    if filtered_records['train']:
        sample = filtered_records['train'][0]
        print(f"示例训练过滤文件: {sample[0]} - 缺失: {', '.join(sample[1])}")
    if filtered_records['val']:
        sample = filtered_records['val'][0]
        print(f"示例验证过滤文件: {sample[0]} - 缺失: {', '.join(sample[1])}")
    print("=" * 50 + "\n")

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L


# 使用示例
if __name__ == "__main__":
    data_path = r"F:\HCDatasets/"
    dataloader(data_path)