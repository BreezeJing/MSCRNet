import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import cv2
from models import *

# 固定参数设定
KITTI = '2015'
loadmodel = r'G:\mobelnetV3\241222aqy\best_3px.tar'
leftimg_path = r'G:\mobelnetV3\data_scene_flow\testing\image_2\000000_10.png'
rightimg_path = r'G:\mobelnetV3\data_scene_flow\testing\image_3\000000_10.png'
model_type = 'stackhourglass'
max_disp = 192
use_cuda = torch.cuda.is_available()
random_seed = 1

# 设定随机种子
torch.manual_seed(random_seed)
if use_cuda:
    torch.cuda.manual_seed(random_seed)

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

# 转换和测试函数
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
        disp1, disp2, disp = model(imgL, imgR)

    disp1 = torch.squeeze(disp1)
    disp2 = torch.squeeze(disp2)
    disp = torch.squeeze(disp)
    return disp1.data.cpu().numpy(),disp2.data.cpu().numpy(),disp.data.cpu().numpy()


# 主函数
def main():
    imgL_o = Image.open(leftimg_path).convert('RGB')
    imgR_o = Image.open(rightimg_path).convert('RGB')

    # imgL_o = imgL_o.resize((1280, 720))
    # imgR_o = imgR_o.resize((1280, 720))
    # imgL_o = imgL_o.resize((960, 512))
    # imgR_o = imgR_o.resize((960, 512))
    # imgL_o = imgL_o.resize((int(2208), int(1216)))
    # imgR_o = imgR_o.resize((int(2208), int(1216)))
    # imgL_o = imgL_o.resize((1216, 352))
    # imgR_o = imgR_o.resize((1216, 352))
    # imgL_o = imgL_o.resize((1536, 2048))
    # imgR_o = imgR_o.resize((1536, 2048))

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)

    start_time = time.time()
    disp1,disp2,pred_disp = test(imgL, imgR)
    print('检测时间: %.2f 秒' % (time.time() - start_time))

    depth = (2873.41333122683 * 18.8689953791335 / 2) / pred_disp
    # DEPTH= cv2.resize(depth,(3072,4096))
    depth_img = Image.fromarray(depth.astype(np.uint16))
    depth_img.save('depth_img.png')

    img = Image.fromarray((pred_disp * 256).astype(np.uint16))
    img.save('Test_disparity.png')

    disparity_color1 = cv2.applyColorMap(cv2.convertScaleAbs(disp1, alpha=256 / max_disp), cv2.COLORMAP_JET)
    disparity_color2 = cv2.applyColorMap(cv2.convertScaleAbs(disp2, alpha=256 / max_disp), cv2.COLORMAP_JET)
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(pred_disp, alpha=256 / max_disp), cv2.COLORMAP_JET)
    cv2.imwrite('disp1.png', disparity_color1)
    cv2.imwrite('disp2.png', disparity_color2)
    cv2.imwrite('disparity_color.png', disparity_color)


if __name__ == '__main__':
    for _ in range(2):
        main()
