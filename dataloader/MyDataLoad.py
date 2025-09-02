import torch.utils.data as data
import random
from PIL import Image, ImageEnhance
import numpy as np
from dataloader import preprocess

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, dis, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.dis = dis
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.transform = preprocess.get_transform(augment=False)

    def __getitem__(self, index):
        left_path = self.left[index]
        right_path = self.right[index]
        dis_path = self.dis[index]

        left_img = self.loader(left_path)
        right_img = self.loader(right_path)
        dis_s = self.dploader(dis_path)

        w, h = left_img.size
        th, tw = 256, 512

        if self.training:
            max_disparity_shift = 10  # 最大视差偏移像素
            attempt = 0
            max_attempts = 50  # 最大尝试次数

            while attempt < max_attempts:
                attempt += 1
                disparity_shift = random.randint(-max_disparity_shift, max_disparity_shift)

                # 根据视差偏移调整裁剪位置
                if disparity_shift >= 0:
                    x1 = random.randint(0, w - tw - disparity_shift)
                    x1_right = x1 + disparity_shift
                else:
                    x1 = random.randint(-disparity_shift, w - tw)
                    x1_right = x1 + disparity_shift  # disparity_shift为负值

                y1 = random.randint(0, h - th)

                # 裁剪左图像
                left_crop = left_img.crop((x1, y1, x1 + tw, y1 + th))
                # 裁剪右图像，考虑视差偏移
                right_crop = right_img.crop((x1_right, y1, x1_right + tw, y1 + th))
                # 裁剪视差图
                dis_crop = dis_s.crop((x1, y1, x1 + tw, y1 + th))

                # 转换视差图为numpy数组并应用视差偏移
                dis_np = np.array(dis_crop, dtype=np.float32) / 256.0
                # 检查裁剪后的视差图是否全为零
                if np.count_nonzero(dis_np) != 0:
                    # 不全为零，跳出循环
                    break
                else:
                    # 裁剪结果全为零，重新尝试
                    continue
                dis_np[dis_np > 0] += disparity_shift


            else:
                # 如果尝试多次仍未找到合适的裁剪，使用默认裁剪（可根据需要调整）
                x1 = (w - tw) // 2
                y1 = (h - th) // 2
                x1_right = x1 + disparity_shift

                left_crop = left_img.crop((x1, y1, x1 + tw, y1 + th))
                right_crop = right_img.crop((x1_right, y1, x1_right + tw, y1 + th))
                dis_crop = dis_s.crop((x1, y1, x1 + tw, y1 + th))

                dis_np = np.array(dis_crop, dtype=np.float32) / 256.0
                dis_np[dis_np > 0] += disparity_shift

            # 更新图像和视差图
            left_img = left_crop
            right_img = right_crop
            dis_s = dis_np

            # 图像色度调整
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Color(left_img)
                left_img = enhancer.enhance(factor)
                enhancer = ImageEnhance.Color(right_img)
                right_img = enhancer.enhance(factor)

            # 添加噪声
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, (th, tw, 3))
                left_img_np = np.array(left_img) / 255.0
                left_img_np += noise
                left_img_np = np.clip(left_img_np, 0, 1)
                left_img = Image.fromarray((left_img_np * 255).astype(np.uint8))

                noise = np.random.normal(0, 0.02, (th, tw, 3))
                right_img_np = np.array(right_img) / 255.0
                right_img_np += noise
                right_img_np = np.clip(right_img_np, 0, 1)
                right_img = Image.fromarray((right_img_np * 255).astype(np.uint8))

            # 转换图像
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        else:
            # 测试模式
            left_img = left_img.crop((0, h - 256, 1024, h))
            right_img = right_img.crop((0, h - 256, 1024, h))
            dis_s = dis_s.crop((0, h - 256, 1024, h))
            dis_s = np.array(dis_s, dtype=np.float32) / 256.0

            # 转换图像
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return left_img, right_img, dis_s

    def __len__(self):
        return len(self.left)
