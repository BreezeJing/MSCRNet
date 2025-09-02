import os
from PIL import Image
import numpy as np

# 定义主文件夹路径
main_folder = r'Z:\calibration\20241104JR-1-Kinect\train_data/'
folders = [os.path.join(main_folder, sub_folder) for sub_folder in ['Left', 'Right', 'dis']]

# 遍历每个文件夹
for folder in folders:
    # 创建输出文件夹
    output_folder = os.path.join(main_folder, os.path.basename(folder) + '_stereo1280')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图像
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 打开图像
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)

            # 将图像缩放为原来的一半
            new_size = (img.width // 2, img.height // 2)
            img_resized = img.resize(new_size, Image.ANTIALIAS)

            # 如果是视差图，需要将图像像素值除以2
            if folder.endswith('dis'):
                img_array = np.array(img_resized, dtype=np.float32) / 2.0
                img_resized = Image.fromarray(np.uint16(img_array))

            # 保存处理后的图像
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

print("所有图像已处理完成！")
