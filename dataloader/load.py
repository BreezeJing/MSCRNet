import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    left_fold  = 'Left_stereo1280/'
    right_fold = 'Right_stereo1280/'
    dis_fold = 'dis_stereo1280/'




    image = [img for img in os.listdir(filepath + left_fold)]

    train = image[:460]
    val   = image[460:480]

    left_train  = [filepath+left_fold+img for img in train]
    right_train = [filepath + right_fold + img for img in train]
    dis_train = [filepath+dis_fold+img for img in train]


    left_val  = [filepath+left_fold+img for img in val]
    right_val = [filepath + right_fold + img for img in val]
    dis_val = [filepath+dis_fold+img for img in val]


    return left_train, right_train, dis_train, left_val, right_val, dis_val