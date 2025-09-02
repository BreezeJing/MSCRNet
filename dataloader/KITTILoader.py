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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, dis, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.dis = dis

        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        dis = self.dis[index]


        left_img = self.loader(left)
        right_img = self.loader(right)
        dis_s=self.dploader(dis)




        if self.training:  
           w, h = left_img.size
           # th, tw = 256, 512
           th, tw = 256, 736
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
           dis_s = np.array(dis_s, dtype=np.float32) / 256
           dis_s = dis_s[int(y1): int(y1 + th), int(x1):int(x1 + tw)]



           processed = preprocess.get_transform(augment=False)
           left_img   = processed(left_img)
           processed = preprocess.get_transform(augment=False)
           right_img = processed(right_img)

           return left_img, right_img, dis_s
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1024, h-256, w, h))
           # print(left_img.size)
           right_img = right_img.crop((w-1024, h-256, w, h))
           w1, h1 = left_img.size

           dis_s = dis_s.crop((int(w - 1024), int(h - 256), int(w), int(h)))
           dis_s = np.ascontiguousarray(dis_s, dtype=np.float32) / 256
           processed = preprocess.get_transform(augment=False)
           left_img = processed(left_img)
           right_img = processed(right_img)


           return left_img, right_img,dis_s

    def __len__(self):
        return len(self.left)
