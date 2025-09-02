import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from dataloader import HCDataloader

def check_stereo_disparity_consistency(left_paths, right_paths, disp_paths):
    """
    执行以下检查：
    1. 文件存在性检查
    2. 图像完整性验证
    3. 双目图像尺寸一致性
    4. 视差图与左图尺寸匹配
    5. 视差值合理性检查
    """
    error_log = {
        'missing_files': 0,
        'corrupted_images': 0,
        'size_mismatch': 0,
        'disparity_issues': 0
    }

    for idx in tqdm(range(len(left_paths)), desc="验证数据"):
        # 获取当前样本路径
        left_path = left_paths[idx]
        right_path = right_paths[idx]
        disp_path = disp_paths[idx]

        # 检查文件存在性 -------------------------------------------------
        missing = []
        if not os.path.exists(left_path):
            missing.append(f"左图: {os.path.basename(left_path)}")
        if not os.path.exists(right_path):
            missing.append(f"右图: {os.path.basename(right_path)}")
        if not os.path.exists(disp_path):
            missing.append(f"视差图: {os.path.basename(disp_path)}")

        if missing:
            print(f"\n文件缺失: {', '.join(missing)}")
            print(f"样本索引: {idx}")
            error_log['missing_files'] += 1
            continue

        # 验证图像完整性 -------------------------------------------------
        try:
            with Image.open(left_path) as img:
                img.verify()
                left_size = img.size
        except Exception as e:
            print(f"\n损坏的左图: {left_path} | 错误: {str(e)}")
            error_log['corrupted_images'] += 1

        try:
            with Image.open(right_path) as img:
                img.verify()
                right_size = img.size
        except Exception as e:
            print(f"\n损坏的右图: {right_path} | 错误: {str(e)}")
            error_log['corrupted_images'] += 1

        try:
            with Image.open(disp_path) as img:
                img.verify()
                disp_size = img.size
                disp_mode = img.mode  # 视差图应为单通道
        except Exception as e:
            print(f"\n损坏的视差图: {disp_path} | 错误: {str(e)}")
            error_log['corrupted_images'] += 1

        # 检查尺寸一致性 -------------------------------------------------
        if left_size != right_size:
            print(f"\n双目尺寸不匹配: 左图 {left_size} vs 右图 {right_size}")
            print(f"左图路径: {left_path}")
            error_log['size_mismatch'] += 1

        if left_size != disp_size:
            print(f"\n视差尺寸不匹配: 左图 {left_size} vs 视差图 {disp_size}")
            print(f"视差图路径: {disp_path}")
            error_log['size_mismatch'] += 1

        # 检查视差值合理性 -------------------------------------------------
        try:
            disp_img = Image.open(disp_path)
            if disp_img.mode not in ['L', 'I', 'F']:
                print(f"\n非常规视差图模式: {disp_mode} (路径: {disp_path})")
                error_log['disparity_issues'] += 1

            # disp_array = np.array(disp_img)
            # if disp_array.dtype == np.uint16:
            #     valid_range = (0, 2 ** 16 - 1)
            # elif disp_array.dtype == np.float32:
            #     valid_range = (0.0, 256.0)  # 根据实际数据调整
            # else:
            #     valid_range = (0, 255)
            #
            # invalid_pixels = np.logical_or(
            #     disp_array < valid_range[0],
            #     disp_array > valid_range[1]
            # ).sum()

            # if invalid_pixels > 0:
            #     print(f"\n异常视差值: {invalid_pixels} 个像素超出范围 {valid_range}")
            #     print(f"视差图路径: {disp_path}")
            #     error_log['disparity_issues'] += 1

        except Exception as e:
            print(f"\n视差图解析失败: {disp_path} | 错误: {str(e)}")
            error_log['disparity_issues'] += 1

    # 生成报告 ----------------------------------------------------------
    print("\n验证报告:")
    print(f"总样本数: {len(left_paths)}")
    print(f"缺失文件: {error_log['missing_files']}")
    print(f"损坏图像: {error_log['corrupted_images']}")
    print(f"尺寸不匹配: {error_log['size_mismatch']}")
    print(f"视差问题: {error_log['disparity_issues']}")

    return error_log


# 使用示例 --------------------------------------------------------------
if __name__ == "__main__":
    # 加载数据路径
    filepath = r"F:\HCDatasets/"
    left_train, right_train, disp_train, left_val, right_val, disp_val = HCDataloader.dataloader(filepath)

    # 检查训练集
    print("正在验证训练集...")
    train_errors = check_stereo_disparity_consistency(left_train, right_train, disp_train)

    # 检查验证集
    print("\n正在验证验证集...")
    val_errors = check_stereo_disparity_consistency(left_val, right_val, disp_val)