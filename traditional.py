import cv2
import numpy as np
# 读取左右彩色图像
left_image = cv2.imread('./resizel.png')
right_image = cv2.imread('./resizer.png')

# 转换为灰度图像
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# 创建SGBM对象
window_size = 3
min_disp = 0
num_disp = 64
sgbm = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size)

# 计算视差图
disparity = sgbm.compute(left_gray, right_gray)

# 通过右图生成右视差图
right_disparity = sgbm.compute(right_gray, left_gray)

# 定义一致性检查阈值
consistency_threshold = 1  # 根据实际情况调整阈值

# 执行一致性检查
consistent_disparity = np.zeros_like(disparity)
for y in range(disparity.shape[0]):
    for x in range(disparity.shape[1]):
        if x - disparity[y, x] >= 0 and x - disparity[y, x] < right_disparity.shape[1]:
            if abs(disparity[y, x] - right_disparity[y, x - disparity[y, x]]) <= consistency_threshold:
                consistent_disparity[y, x] = disparity[y, x]


# 根据视差图计算深度图
depth = 26644 / disparity

# 显示视差图和深度图
cv2.imshow('Disparity Map', disparity*256)
cv2.imshow('Depth Map', depth)

cv2.waitKey(0)
cv2.destroyAllWindows()