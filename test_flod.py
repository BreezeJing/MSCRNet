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
left_img_folder = r'G:\mobelnetV4\grapedata\left'  # 左图文件夹路径
right_img_folder = r'G:\mobelnetV4\grapedata\right'  # 右图文件夹路径
output_folder = r'G:\mobelnetV4\grapedata/depth'  # 输出文件夹路径
model_type = 'stackhourglass'
max_disp = 192
use_cuda = torch.cuda.is_available()
loadmodel = r'G:\mobelnetV4\241227scenseflow\best_3px.tar'

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

    for left_img_path, right_img_path in zip(left_images, right_images):
        imgL_o = Image.open(left_img_path).convert('RGB')
        imgR_o = Image.open(right_img_path).convert('RGB')

        # imgL_o = imgL_o.resize((1280, 720))
        # imgR_o = imgR_o.resize((1280, 720))
        imgL_o = imgL_o.resize((1280, 736))
        imgR_o = imgR_o.resize((1280, 736))
        # imgL_o = imgL_o.resize((int(2208), int(1216)))
        # imgR_o = imgR_o.resize((int(2208), int(1216)))
        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        start_time = time.time()
        pred_disp = test(imgL, imgR)
        print(f'检测时间: {time.time() - start_time:.2f} 秒，处理图片: {left_img_path}')

        file_base_name = os.path.basename(left_img_path).split('.')[0]
        depth = (532.275 * 119.961) / (pred_disp+4)
        # depth = (1046.20510239201 * 25.6900852509765/2) / pred_disp
        DEPTH = cv2.resize(depth, (2208, 1242))
        depth_img = Image.fromarray(DEPTH.astype(np.uint16))
        depth_img.save(os.path.join(output_folder, f'{file_base_name}_depth.png'))

        img = Image.fromarray((pred_disp * 256).astype(np.uint16))
        img.save(os.path.join(output_folder, f'{file_base_name}_disparity.png'))

        disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(pred_disp, alpha=256 / max_disp), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_folder, f'{file_base_name}_disparity_color.png'), disparity_color)

if __name__ == '__main__':
    main()
