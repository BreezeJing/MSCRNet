import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import cv2
from models import anet, basic  # 确保 models.py 包含 anet 和 basic 模型定义

# 兼容不同版本的 Pillow
try:
    from PIL import Resampling
    resample_method = Resampling.BILINEAR
except ImportError:
    resample_method = Image.BILINEAR
def parse_arguments():
    parser = argparse.ArgumentParser(description='Stereo Image Depth Estimation with Dynamic Resolution Handling')
    parser.add_argument('--left_folder', type=str, default=r'G:\zed0904\left', help='左图文件夹路径')
    parser.add_argument('--right_folder', type=str, default=r'G:\zed0904\right', help='右图文件夹路径')
    parser.add_argument('--output_folder', type=str, default=r'G:\zed0904/depth_disp/', help='输出文件夹路径')
    parser.add_argument('--model_type', type=str, default='stackhourglass', choices=['stackhourglass', 'basic'],
                        help='模型类型')
    parser.add_argument('--max_disp', type=int, default=192, help='最大视差')
    parser.add_argument('--model_path', type=str, default=r'./1_2best_loss.tar', help='预训练模型路径')
    parser.add_argument('--network_resolution', type=str, default='1280x720', help='网络输入分辨率，例如1280x720')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小（当前代码为单张处理）')
    return parser.parse_args()


def initialize_model(model_type, max_disp, use_cuda, loadmodel):
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
        raise ValueError(f'未知的模型类型: {model_type}')

    model = nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()

    if loadmodel:
        print('加载 PSMNet 模型')
        state_dict = torch.load(loadmodel, map_location='cuda' if use_cuda else 'cpu')
        model.load_state_dict(state_dict['state_dict'])

    print(f'模型参数数量: {sum(p.data.nelement() for p in model.parameters())}')
    return model


def get_transform():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**normal_mean_var)
    ])


def test(model, imgL, imgR, device):
    model.eval()
    imgL, imgR = imgL.to(device), imgR.to(device)
    imgL, imgR = imgL.unsqueeze(0), imgR.unsqueeze(0)

    with torch.no_grad():
        _, _, disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    return disp.cpu().numpy()


def save_outputs(pred_disp, file_base_name, output_folder, max_disp, original_size, depth_scale=24.826*1015/2):
    original_width, original_height = original_size

    # 调整视差图回原始尺寸
    disp_resized = cv2.resize(pred_disp, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    # 保存深度图
    depth = depth_scale / (disp_resized + 1e-6)  # 避免除以零
    depth = np.clip(depth, 0, 65535)  # 防止溢出
    depth_img = Image.fromarray(depth.astype(np.uint16))
    depth_img.save(os.path.join(output_folder, f'{file_base_name}_depth.png'))

    # 保存视差图
    disparity = (disp_resized * 256).astype(np.uint16)
    disp_img = Image.fromarray(disparity)
    disp_img.save(os.path.join(output_folder, f'{file_base_name}_disparity.png'))

    # 保存彩色视差图
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_resized, alpha=255 / max_disp), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_folder, f'{file_base_name}_disparity_color.png'), disparity_color)


def main():
    args = parse_arguments()

    # 解析网络输入分辨率
    try:
        net_width, net_height = map(int, args.network_resolution.lower().split('x'))
    except:
        raise ValueError('网络分辨率格式错误，应为宽x高，例如1280x720')

    # 确保输出文件夹存在
    os.makedirs(args.output_folder, exist_ok=True)

    # 检查CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # 初始化模型
    model = initialize_model(args.model_type, args.max_disp, use_cuda, args.model_path)

    # 图像转换
    infer_transform = get_transform()

    # 预加载图像列表
    left_images = sorted(
        [file for file in os.listdir(args.left_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    right_images = sorted(
        [file for file in os.listdir(args.right_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(left_images) != len(right_images):
        raise ValueError('左图和右图的数量不一致')

    total_images = len(left_images)
    print(f'总共需要处理 {total_images} 对图像')

    for idx, (left_file, right_file) in enumerate(zip(left_images, right_images), 1):
        left_img_path = os.path.join(args.left_folder, left_file)
        right_img_path = os.path.join(args.right_folder, right_file)

        # 打开原始图像并记录原始尺寸
        imgL_original = Image.open(left_img_path).convert('RGB')
        imgR_original = Image.open(right_img_path).convert('RGB')
        original_size = imgL_original.size  # (width, height)

        # 调整大小到网络输入尺寸，使用兼容的重采样方法
        imgL_resized = imgL_original.resize((net_width, net_height), resample=resample_method)
        imgR_resized = imgR_original.resize((net_width, net_height), resample=resample_method)

        # 转换为张量
        imgL = infer_transform(imgL_resized)
        imgR = infer_transform(imgR_resized)

        # 模型推理
        start_time = time.time()
        pred_disp = test(model, imgL, imgR, device)
        elapsed_time = time.time() - start_time
        print(f'[{idx}/{total_images}] 处理 {left_file} 耗时: {elapsed_time:.2f} 秒')

        # 保存输出
        file_base_name = os.path.splitext(left_file)[0]
        save_outputs(pred_disp, file_base_name, args.output_folder, args.max_disp, original_size)

    print('所有图像处理完成')


if __name__ == '__main__':
    main()
