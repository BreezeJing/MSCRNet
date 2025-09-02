import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = r'Z:\calibration\20241104JR-1-Kinect\train_data\dis'
output_folder = r'Z:\calibration\20241104JR-1-Kinect\train_data\dis_stereo1280'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):  # 确保只处理PNG文件
        input_path = os.path.join(input_folder, filename)

        # 读取原始16位视差图
        disparity_map = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if disparity_map is None:
            print(f"无法读取图像: {input_path}")
            continue

        # 将视差图缩小为原来的一半
        height, width = disparity_map.shape
        resized_disparity_map = cv2.resize(disparity_map, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)

        # 视差值除以2
        resized_disparity_map = resized_disparity_map / 2.0

        # 转换为16位无符号整数
        resized_disparity_map = resized_disparity_map.astype(np.uint16)

        # 保存结果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_disparity_map)

        print(f"处理完成: {filename}")

print("批量处理完成。")
