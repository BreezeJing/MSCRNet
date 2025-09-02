import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import time
import numpy as np
import cv2
from models import *  # 请确保正确导入模型

# 固定参数配置
KITTI = '2015'
loadmodel = r'E:\Code\Stereo\250318HCDatasets\best_loss.pth'
# leftimg_path = r'G:\code\local\zed\F/left.png'
# rightimg_path = r'G:\code\local\zed\F/right.png'
leftimg_path = r'C:\Users\jingx\Desktop\img_test\left\1.jpeg'
rightimg_path = r'C:\Users\jingx\Desktop\img_test\right/1.jpeg'
model_type = 'stackhourglass'
max_disp = 192
use_cuda = torch.cuda.is_available()
random_seed = 1

# 设置随机种子
torch.manual_seed(random_seed)
if use_cuda:
    torch.cuda.manual_seed(random_seed)


# 模型初始化
def create_model():
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
    return model


model = create_model()
if use_cuda:
    model = model.cuda()

# 加载模型部分
if loadmodel is not None:
    print('加载 PSMNet 模型')
    state_dict = torch.load(loadmodel)

    # 修复键名：去除 "module." 前缀
    new_state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict["state_dict"].items()
    }

    # 加载修复后的权重
    model.load_state_dict(new_state_dict, strict=True)
    model = nn.DataParallel(model)
    print('模型权重加载成功！')

# 图像预处理
normal_mean_var = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**normal_mean_var)
])


def pad_image(image, multiple=32):
    """将图像填充至32的倍数尺寸"""
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    padding = (0, 0, new_w - w, new_h - h)  # 只在右侧和底部填充
    return ImageOps.expand(image, padding, fill=0), (w, h)


def process_disparity(disp, orig_size):
    """视差后处理"""
    # 裁剪回原始尺寸
    disp = disp[:orig_size[1], :orig_size[0]]

    # 去除无效视差
    disp[disp > 256] = 0  # 去除超过最大视差的值
    disp[disp < 0] = 0  # 去除负视差值
    return disp


def inference(image_left, image_right):
    """执行立体匹配推理"""
    # 记录原始尺寸
    orig_size = image_left.size

    # 填充图像
    left_padded, _ = pad_image(image_left)
    right_padded, _ = pad_image(image_right)

    # 转换张量
    left_tensor = transform(left_padded)
    right_tensor = transform(right_padded)

    # 执行推理
    model.eval()
    with torch.no_grad():
        if use_cuda:
            left_tensor = left_tensor.cuda().unsqueeze(0)
            right_tensor = right_tensor.cuda().unsqueeze(0)
        start_time=time.time()
        disp1, disp2, disp = model(left_tensor, right_tensor)
        print('infer_time:',time.time()-start_time)
        # 处理各阶段视差图
        disp1 = process_disparity(disp1.squeeze().cpu().numpy(), orig_size)
        disp2 = process_disparity(disp2.squeeze().cpu().numpy(), orig_size)
        final_disp = process_disparity(disp.squeeze().cpu().numpy(), orig_size)

    return disp1, disp2, final_disp


def generate_depth(disp, focal_length=1087.47099650264, baseline=600):
    """生成深度图"""
    valid_mask = disp > 0
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[valid_mask] = (focal_length * baseline) / disp[valid_mask]
    return depth


def visualize_disparity(disp, max_value=192):
    """生成彩色视差图"""
    disp_normalized = cv2.convertScaleAbs(disp, alpha=255 / max_value)
    return cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)


def main():
    # 加载图像
    imgL = Image.open(leftimg_path).convert('RGB')
    imgR = Image.open(rightimg_path).convert('RGB')
    # imgL = imgL.resize((480, 384))
    # imgR = imgR.resize((480, 384))
    # imgL = imgL.resize((1920, 1080))
    # imgR = imgR.resize((1920, 1080))
    # imgL = imgL.resize((1280, 720))
    # imgR = imgR.resize((1280, 720))
    # imgL = imgL.resize((1024, 640))
    # imgR = imgR.resize((1024, 640))
    # 执行推理
    start_time = time.time()
    disp1, disp2, final_disp = inference(imgL, imgR)
    print(f'处理耗时: {time.time() - start_time:.2f}s')

    # 生成深度图
    depth_map = generate_depth(final_disp)

    # 保存结果
    cv2.imwrite('disp1_color.png', visualize_disparity(disp1))
    cv2.imwrite('disp2_color.png', visualize_disparity(disp2))
    cv2.imwrite('final_disp_color.png', visualize_disparity(final_disp))

    # 保存16位视差图
    cv2.imwrite('final_disp.png', (final_disp * 1000).astype(np.uint16))

    # 保存浮点深度图
    cv2.imwrite('depth_map.png', depth_map.astype(np.uint16))


if __name__ == '__main__':
    for _ in range(1):  # 示例运行两次
        main()