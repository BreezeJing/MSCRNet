import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import time
import numpy as np
import cv2
from models import *  # 请替换为实际模型导入

# 参数配置
left_img_folder = r'E:\ShareWorkSpace\Data\ALG_300_3588\left_rect'
right_img_folder = r'E:\ShareWorkSpace\Data\ALG_300_3588\right_rect'
output_folder = r'E:\ShareWorkSpace\Data\ALG_300_3588\my_depth'
model_type = 'stackhourglass'
max_disp = 192
use_cuda = torch.cuda.is_available()
loadmodel = r'E:\Code\Stereo\250403HCDatasets\best_3px.pth'

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

# 模型初始化
if model_type == 'stackhourglass':
    lsp_channel = 4
    lsp_mode = 'separate'
    lsp_width = 3
    lsp_dilation = [1, 2, 4, 8]
    affinity_settings = {
        'win_w': lsp_width,
        'win_h': lsp_width,
        'dilation': lsp_dilation
    }
    udc = True
    model = anet(max_disp, use_concat_volume=True, struct_fea_c=lsp_channel,
                 fuse_mode=lsp_mode, affinity_settings=affinity_settings, udc=udc)
elif model_type == 'basic':
    model = basic(max_disp)
else:
    raise ValueError("不支持的模型类型")


if use_cuda:
    model = model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练权重
if loadmodel:
    print(f"加载模型权重: {loadmodel}")
    checkpoint = torch.load(loadmodel, map_location=device)
    state_dict = checkpoint['state_dict']
    # 统一处理多GPU参数前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
model = nn.DataParallel(model)
# 图像预处理
normal_mean_var = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**normal_mean_var)
])


def inference_batch(left_tensor, right_tensor):
    model.eval()
    with torch.no_grad():
        if use_cuda:
            left_tensor, right_tensor = left_tensor.cuda(), right_tensor.cuda()
        _, _, disp = model(left_tensor.unsqueeze(0), right_tensor.unsqueeze(0))
    return disp.squeeze().cpu().numpy()


def pad_to_multiple(image, multiple=32):
    """将图像填充至指定倍数的尺寸"""
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    pad_w = new_w - w
    pad_h = new_h - h
    return ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0), (w, h)


def process_image_pair(left_path, right_path):
    # 读取图像并记录原始尺寸
    left_img = Image.open(left_path).convert('RGB')
    right_img = Image.open(right_path).convert('RGB')
    # left_img = left_img.resize((552, 310))
    # right_img = right_img.resize((552, 310))
    left_img = left_img.resize((960, 768))
    right_img = right_img.resize((960, 768))
    # 填充图像至32的倍数
    left_padded, orig_size = pad_to_multiple(left_img)
    right_padded, _ = pad_to_multiple(right_img)

    # 转换张量
    left_tensor = transform(left_padded)
    right_tensor = transform(right_padded)

    # 执行推理
    start_time = time.time()
    disp_map = inference_batch(left_tensor, right_tensor)
    print(f"处理 {os.path.basename(left_path)} 耗时: {time.time() - start_time:.2f}s")

    # 裁剪回原始尺寸
    disp_map = disp_map[:orig_size[1], :orig_size[0]]

    # 去除无效视差
    disp_map[disp_map > max_disp] = 0
    disp_map[disp_map < 0] = 0

    return disp_map


def main():
    # 获取匹配的图像对
    left_images = sorted([os.path.join(left_img_folder, f)
                          for f in os.listdir(left_img_folder) if f.endswith(('.jpg', '.png'))])
    right_images = sorted([os.path.join(right_img_folder, f)
                           for f in os.listdir(right_img_folder) if f.endswith(('.jpg', '.png'))])

    for left_path, right_path in zip(left_images, right_images):
        try:
            # 处理图像对
            disp = process_image_pair(left_path, right_path)

            # 生成深度图（根据相机参数调整）
            depth_map = (1087.47099650264 * 60.0 * 10) / (disp + 1e-6)  # 防止除以零

            # 保存结果
            base_name = os.path.splitext(os.path.basename(left_path))[0]

            # 保存16位视差图
            cv2.imwrite(os.path.join(output_folder, f"{base_name}.png"),
                        (disp * 256).astype(np.uint16))

            # 保存彩色视差图
            disp_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(disp, alpha=255 / max_disp),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_disp_color.jpg"), disp_vis)

            # 保存深度图
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_depth.png"),
                        depth_map.astype(np.uint16))

        except Exception as e:
            print(f"处理 {left_path} 时出错: {str(e)}")


if __name__ == "__main__":
    main()