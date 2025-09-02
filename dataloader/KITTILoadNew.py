import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataloader import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class StereoColorJitter(object):
    """立体图像颜色抖动增强（修复版）"""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, asymmetric_prob=0.5):
        self.asymmetric_prob = asymmetric_prob
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue / 3.14  # 正确范围调整
        )

    def __call__(self, left_img, right_img):
        """处理左右图像对（修复分割错误）"""
        if random.random() < self.asymmetric_prob:
            # 非对称增强保持不变
            return self.color_jitter(left_img), self.color_jitter(right_img)
        else:
            # 修复关键步骤：使用numpy进行空间分割
            # 转换为numpy数组并垂直堆叠
            stacked = np.vstack([np.array(left_img), np.array(right_img)])
            # 应用颜色抖动
            stacked_img = self.color_jitter(Image.fromarray(stacked))
            # 转换回numpy进行分割
            stacked_np = np.array(stacked_img)
            h = left_img.height  # 原始单图高度
            # 按高度分割回左右图像
            return (
                Image.fromarray(stacked_np[:h]),
                Image.fromarray(stacked_np[h:])
            )

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, dis, training,
                loader=default_loader, dploader=disparity_loader,
                color_aug_params={
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                    'hue': 0.1,
                    'asymmetric_prob': 0.5
                }):
        self.left = left
        self.right = right
        self.dis = dis

        self.loader = loader
        self.dploader = dploader
        self.training = training
        # 初始化颜色增强
        if training:
            self.color_aug = StereoColorJitter(**color_aug_params)
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        dis = self.dis[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dis_s = self.dploader(dis)
        # print(left, right, dis)
        if self.training:
            w, h = left_img.size
            # th, tw = 256, 512
            th, tw = 256, 736
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            left_img, right_img = self.color_aug(left_img, right_img)
            dis_s = np.array(dis_s, dtype=np.float32) / 256
            dis_s = dis_s[int(y1): int(y1 + th), int(x1):int(x1 + tw)]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            processed = preprocess.get_transform(augment=False)
            right_img = processed(right_img)

            return left_img, right_img, dis_s
        else:
            # 测试时32倍数对齐
            w, h = left_img.size

            # 计算填充量
            pad_w = (32 - (w % 32)) % 32
            pad_h = (32 - (h % 32)) % 32

            # 创建新图像并填充
            def pad_image(img):
                return ImageOps.expand(img,
                                       border=(0, 0, pad_w, pad_h),
                                       fill=0)

            left_img = pad_image(left_img)
            right_img = pad_image(right_img)

            # 视差图处理
            dis_s = np.array(dis_s, dtype=np.float32) / 256
            dis_s = np.pad(dis_s,
                           ((0, pad_h), (0, pad_w)),
                           mode='constant'
                           )
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dis_s

    def __len__(self):
        return len(self.left)
