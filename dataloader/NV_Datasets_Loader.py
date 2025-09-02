import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps

from . import preprocess
from . import enhance

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)

def default_loader(path):
    """读取 RGB 图像"""
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    """读取 uint8 三通道编码的视差／深度图"""
    return Image.open(path)

def depth_uint8_decoding(depth_uint8, scale=1000):
    """
    将 uint8 三通道编码解码为浮点深度：
      D = (R*255*255 + G*255 + B) / scale
    """
    arr = depth_uint8.astype(np.float32)
    decoded = arr[...,0] * 255 * 255 + arr[...,1] * 255 + arr[...,2]
    return decoded / scale

class myImageFloder(data.Dataset):
    def __init__(self, left_list, right_list, disp_list, training,
                 loader=default_loader, dploader=disparity_loader):
        """
        left_list, right_list, disp_list: 三个等长的文件路径列表
        training: True=训练模式（增强），False=测试模式（pad）
        """
        self.left      = left_list
        self.right     = right_list
        self.disp      = disp_list
        self.loader    = loader
        self.dploader  = dploader
        self.training  = training
        self.to_tensor = preprocess.get_transform(augment=False)

    def __len__(self):
        return len(self.left)

    def __getitem__(self, idx):
        # 1. 读取 RGB 和 uint8 深度图
        left_img  = self.loader(self.left[idx])
        right_img = self.loader(self.right[idx])
        disp_img  = self.dploader(self.disp[idx])

        # 2. 解码 uint8 深度到浮点 ndarray
        disp_np = np.array(disp_img, dtype=np.uint8)           # H×W×3
        depth_f = depth_uint8_decoding(disp_np)                # H×W

        if self.training:
            # ---- 训练：增强 ----
            left_img, right_img, depth_f, _ = enhance.enhance_img(
                left_img, right_img, depth_f, None
            )
            left_t  = self.to_tensor(left_img)
            right_t = self.to_tensor(right_img)
            disp_t  = torch.from_numpy(depth_f.astype(np.float32))
            return left_t, right_t, disp_t

        else:
            # ---- 测试：pad 到 32 的倍数 ----
            w, h    = left_img.size
            pad_w   = (32 - w % 32) % 32
            pad_h   = (32 - h % 32) % 32

            left_p  = ImageOps.expand(left_img,  border=(0,0,pad_w,pad_h), fill=0)
            right_p = ImageOps.expand(right_img, border=(0,0,pad_w,pad_h), fill=0)
            depth_p = np.pad(depth_f, ((0,pad_h),(0,pad_w)), mode='constant')

            left_t  = self.to_tensor(left_p)
            right_t = self.to_tensor(right_p)
            disp_t  = torch.from_numpy(depth_p.astype(np.float32))
            return left_t, right_t, disp_t
