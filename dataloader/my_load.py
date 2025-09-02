import os


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def dataloader(filepath):
    left_fold = 'Left_stereo1280'
    right_fold = 'Right_stereo1280'
    disp_fold = 'dis_stereo1280'

    # 获取左图像列表并排序
    left_images = [img for img in os.listdir(os.path.join(filepath, left_fold)) if is_image_file(img)]
    left_images.sort()

    # 根据左图像列表生成右图像和视差图列表
    right_images = [img for img in left_images]
    disp_images = [img for img in left_images]

    # 划分训练集和验证集，假设前190张为训练集
    train_size = 460
    left_train = [os.path.join(filepath, left_fold, img) for img in left_images[:train_size]]
    right_train = [os.path.join(filepath, right_fold, img) for img in right_images[:train_size]]
    disp_train = [os.path.join(filepath, disp_fold, img) for img in disp_images[:train_size]]

    left_val = [os.path.join(filepath, left_fold, img) for img in left_images[train_size:]]
    right_val = [os.path.join(filepath, right_fold, img) for img in right_images[train_size:]]
    disp_val = [os.path.join(filepath, disp_fold, img) for img in disp_images[train_size:]]

    return left_train, right_train, disp_train, left_val, right_val, disp_val
