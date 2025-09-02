import torchvision
import numpy as np
import dataloader.readpfm as rp
import dataloader.flow_transforms as flow_transforms
from PIL import Image, ImageOps

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)


def enhance_img(left_img, right_img, dataL, dataR):
    max_h = left_img.size[1]//2
    max_w = left_img.size[0]//2
    max_h, max_w = 256, 512

    # photometric unsymmetric-augmentation
    random_brightness = np.random.uniform(0.4, 1.6,2)
    random_gamma = np.random.uniform(0.4, 1.6,2)
    random_contrast = np.random.uniform(0.4, 1.6,2)

    left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
    left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
    left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
    right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
    right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
    right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
    right_img = np.asarray(right_img)
    left_img = np.asarray(left_img)

    # horizontal flip
    if dataR != None:
        if np.random.binomial(1,0.5):
            tmp = right_img
            right_img = left_img[:,::-1]
            left_img = tmp[:,::-1]
            tmp = dataR
            dataR = dataL[0][:,::-1]
            dataL = tmp[0][:,::-1]

    # geometric unsymmetric-augmentation
    angle=0;px=0
    if np.random.binomial(1,0.5):
        angle=0.1;px=2
    co_transform = flow_transforms.Compose([
        flow_transforms.RandomTranslationTransform((max_h,max_w)),
        flow_transforms.RandomVdisp(angle,px),
        # flow_transforms.Scale(np.random.uniform(0.8, 1.2),order=2),
        flow_transforms.RandomCrop((max_h,max_w)),
        ])
    augmented,dataL = co_transform([left_img, right_img], dataL)
    left_img = augmented[0]
    right_img = augmented[1]

    left_img = left_img.copy()
    right_img = right_img.copy()

    # randomly occlude a region
    if np.random.binomial(1,0.5):
        sx = int(np.random.uniform(50,150))
        sy = int(np.random.uniform(50,150))
        cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
        cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
        right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]
    return Image.fromarray(left_img), Image.fromarray(right_img), dataL, dataR

if __name__ == "__main__":
    a,b,c,d = enhance_img(
        default_loader(r"E:\Datasets\Public\SceneFlow\flyingthings3d__frames_cleanpass\TRAIN\A\0000\left\0006.png"),
        default_loader(r"E:\Datasets\Public\SceneFlow\flyingthings3d__frames_cleanpass\TRAIN\A\0000\right\0006.png"),
        disparity_loader(r"E:\Datasets\Public\SceneFlow\flyingthings3d__disparity\TRAIN\A\0000\left\0006.pfm"),
        disparity_loader(r"E:\Datasets\Public\SceneFlow\flyingthings3d__disparity\TRAIN\A\0000\right\0006.pfm")
        )
    a=Image.fromarray(a)
    a.save("dataloader/img_test/a_5.png")
    