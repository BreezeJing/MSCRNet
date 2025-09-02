import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import cv2
from models import *

# 参数设定
left_img_folder = r'G:\mobelnet\training\image_2'  # 左图文件夹路径
right_img_folder = r'G:\mobelnet\training\image_3'  # 右图文件夹路径
disp_gt_folder = r'G:\mobelnet\training\disp_occ_0'  # 真值视差图文件夹路径
output_folder = r'G:\mobelnet\training\output'  # 输出文件夹路径
model_type = 'stackhourglass'
max_disp = 192
use_cuda = torch.cuda.is_available()
loadmodel = './1_2best_loss.tar'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
    print('没有这种模型类型')
    exit()

model = nn.DataParallel(model)
if use_cuda:
    model = model.cuda()

# 加载模型
if loadmodel is not None:
    print('加载 PSMNet 模型')
    state_dict = torch.load(loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('模型参数数量: {}'.format(sum(p.data.nelement() for p in model.parameters())))

# 图像转换
normal_mean_var = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
infer_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**normal_mean_var)
])

def test(imgL, imgR):
    model.eval()
    if use_cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
    imgL, imgR = imgL.unsqueeze(0), imgR.unsqueeze(0)

    with torch.no_grad():
        _, _, disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    return disp.data.cpu().numpy()

# 主函数
def main():
    # 从文件夹读取所有图片
    left_images = sorted([os.path.join(left_img_folder, file) for file in os.listdir(left_img_folder) if file.endswith('.png')])
    right_images = sorted([os.path.join(right_img_folder, file) for file in os.listdir(right_img_folder) if file.endswith('.png')])

    total_epe = 0
    total_three_pixel_error = 0
    total_valid_pixels = 0

    for left_img_path, right_img_path in zip(left_images, right_images):
        imgL_o = Image.open(left_img_path).convert('RGB')
        imgR_o = Image.open(right_img_path).convert('RGB')

        # 获取图像尺寸
        width, height = imgL_o.size

        # 定义裁剪区域，左下角960x720
        left = 0
        upper = height - 256
        right = 1024
        lower = height

        # 裁剪图像
        imgL_o = imgL_o.crop((left, upper, right, lower))
        imgR_o = imgR_o.crop((left, upper, right, lower))

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        start_time = time.time()
        pred_disp = test(imgL, imgR)
        print(f'检测时间: {time.time() - start_time:.2f} 秒，处理图片: {left_img_path}')

        file_base_name = os.path.basename(left_img_path).split('.')[0]

        # 保存深度和视差图
        depth = (496.555131117712 * 25) / pred_disp
        depth_img = Image.fromarray(depth.astype(np.uint16))
        depth_img.save(os.path.join(output_folder, f'{file_base_name}_depth.png'))

        img = Image.fromarray((pred_disp * 256).astype(np.uint16))
        img.save(os.path.join(output_folder, f'{file_base_name}_disparity.png'))

        disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(pred_disp, alpha=256 / max_disp), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_folder, f'{file_base_name}_disparity_color.png'), disparity_color)

        # 加载真值视差图
        disp_gt_path = os.path.join(disp_gt_folder, os.path.basename(left_img_path))
        disp_gt = cv2.imread(disp_gt_path, cv2.IMREAD_UNCHANGED)

        if disp_gt is None:
            print(f'无法加载真值视差图：{disp_gt_path}')
            continue

        # 裁剪真值视差图
        disp_gt = disp_gt.astype(np.float32)
        disp_gt = disp_gt[upper:lower, left:right]

        # 如果真值视差图被放大了256倍，需要缩放回来
        disp_gt = disp_gt / 256.0

        # 确保数据类型为float32
        pred_disp = pred_disp.astype(np.float32)

        # 创建真值有效掩码
        valid_mask = disp_gt > 0

        # 计算绝对误差
        abs_diff = np.abs(pred_disp - disp_gt)[valid_mask]

        # 计算EPE误差
        EPE = np.mean(abs_diff)

        # 计算3像素误差
        three_pixel_error = np.mean(abs_diff > 3.0)

        print(f'图片 {file_base_name}: EPE = {EPE:.4f}, 3像素误差 = {three_pixel_error*100:.2f}%')

        # 累计误差
        total_epe += np.sum(abs_diff)
        total_three_pixel_error += np.sum(abs_diff > 3.0)
        total_valid_pixels += np.sum(valid_mask)

    # 计算整个数据集的平均误差
    if total_valid_pixels > 0:
        avg_epe = total_epe / total_valid_pixels
        avg_three_pixel_error = total_three_pixel_error / total_valid_pixels

        print(f'数据集平均 EPE: {avg_epe:.4f}')
        print(f'数据集平均 3像素误差: {avg_three_pixel_error*100:.2f}%')
    else:
        print('没有有效的真值像素用于计算误差。')

if __name__ == '__main__':
    main()
