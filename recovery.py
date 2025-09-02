import cv2
import numpy as np

#######################
# 用户给定的参数 (示例)
#######################
# 来自 undistort_images 使用的单目参数
mono_K_left = np.array([[998.985779977150, 0, 1281.47044851211],
                        [0, 997.847449567653, 711.547304195628],
                        [0, 0, 1]])
mono_dist_left = np.array([-0.0204847432227852, -0.0140691326657714, 0, 0, 0])

# 来自 stereo_rectify_images 使用的参数（以左相机为例）
stereo_K_left = np.array([[997.332185459066, 0, 1281.69769814937],
                          [0, 996.674963810831, 711.551841672795],
                          [0, 0, 1]])
dist_left = np.array([-0.0249904028217554, -0.00818355353490463, 0, 0, 0, 0])
# R1, P1 同样来自立体校正结果（需真实参数）
R1 = np.array([[0.999998731341511, 0.000423071091410169, -0.00153568428401341],
               [-0.000428110559412949, 0.999994520173932, -0.00328273414345566],
               [0.00153428703881376, 0.00328338742144503, 0.999993432643597]])
P1 = np.array([[997.332185459066, 0, 1281.69769814937, 0],
               [0, 996.674963810831, 711.551841672795, 0],
               [0, 0, 1, 0]])

# 图像分辨率（需与实际情况匹配）
width, height = 1920, 1080

#######################
# 重投影与畸变函数定义
#######################

def distort_points(x_und, y_und, dist_coeffs):
    """
    根据给定的畸变参数将无畸变的归一化坐标 (x_und, y_und) 转换为畸变坐标 (x_dist, y_dist)。
    dist_coeffs = [k1, k2, p1, p2, k3]
    对于本例，p1, p2=0，k3=0或未使用。
    """
    k1, k2, p1, p2, k3 = 0, 0, 0, 0, 0
    if len(dist_coeffs) >= 1:
        k1 = dist_coeffs[0]
    if len(dist_coeffs) >= 2:
        k2 = dist_coeffs[1]
    if len(dist_coeffs) >= 3:
        p1 = dist_coeffs[2]
    if len(dist_coeffs) >= 4:
        p2 = dist_coeffs[3]
    if len(dist_coeffs) >= 5:
        k3 = dist_coeffs[4]

    r2 = x_und*x_und + y_und*y_und
    # 径向畸变
    radial = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)
    # 切向畸变
    x_dist = x_und*radial + 2*p1*x_und*y_und + p2*(r2 + 2*x_und*x_und)
    y_dist = y_und*radial + p1*(r2 + 2*y_und*y_und) + 2*p2*x_und*y_und

    return x_dist, y_dist

def invert_rectification(u_rect, v_rect, R1, P1, stereo_K):
    """
    将校正图像坐标 (u_rect, v_rect) 转换回无畸变图像坐标系下的归一化坐标。
    P1 = stereo_K * [R1|t], 这里假设 t 很小或者忽略平移对归一化影响(具体情况需严格考虑)。
    我们可通过如下步骤：
    1. 归一化：x_rect = (u_rect - cx_rect)/fx_rect, y_rect = (v_rect - cy_rect)/fy_rect
       (cx_rect, cy_rect, fx_rect, fy_rect) 来自 P1 或 stereo_K
    2. 使用 R1 的逆（R1^T）将 (x_rect, y_rect) 转换回原始无畸变坐标系
    """
    fx_rect = P1[0,0]
    fy_rect = P1[1,1]
    cx_rect = P1[0,2]
    cy_rect = P1[1,2]

    x_rect = (u_rect - cx_rect)/fx_rect
    y_rect = (v_rect - cy_rect)/fy_rect

    # 将rect坐标系下的点映射回原始坐标系，需要使用R1的逆
    # 无畸变坐标系下的点（X_und, Y_und, Z_und=1）
    # R1为3x3旋转矩阵，将 (x_und, y_und, 1) -> (x_rect, y_rect, 1)
    # 所以 (x_und, y_und, 1) = R1^T*(x_rect, y_rect, 1)
    inv_R1 = R1.T
    vec_rect = np.array([x_rect, y_rect, 1.0])
    vec_und = inv_R1 @ vec_rect
    # 归一化，让Z=1
    vec_und /= vec_und[2]

    x_und = vec_und[0]
    y_und = vec_und[1]

    return x_und, y_und

def normalized_to_pixel(x, y, K):
    """
    将归一化坐标 (x, y) 转换到像素坐标
    K 为相机内参矩阵
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    u = x*fx + cx
    v = y*fy + cy
    return u, v

#######################
# 逆向恢复主函数
#######################
def recover_original_image(rectified_image, R1, P1, stereo_K_left, dist_left, mono_K_left, mono_dist_left):
    """
    尝试将校正后的图像逆向恢复为原始图像的近似值。
    rectified_image：校正后的左图像
    R1, P1, stereo_K_left, dist_left：立体校正时使用的参数
    mono_K_left, mono_dist_left：单目去畸变使用的参数
    """
    h, w = rectified_image.shape[:2]

    # 创建一个空白图像，用于存放恢复结果
    recovered = np.zeros_like(rectified_image)

    # 遍历校正后的每个像素
    for v_rect in range(h):
        for u_rect in range(w):
            # 1. 从rect坐标 -> 无畸变坐标
            x_und, y_und = invert_rectification(u_rect, v_rect, R1, P1, stereo_K_left)

            # 此时 (x_und, y_und) 是无畸变归一化坐标，需要加上单目畸变回去以找到原始点
            # 2. 将无畸变坐标重新添加畸变
            x_dist, y_dist = distort_points(x_und, y_und, mono_dist_left)

            # 3. 使用原始内参将归一化坐标转换为像素坐标
            u_orig, v_orig = normalized_to_pixel(x_dist, y_dist, mono_K_left)

            # 在原始图像坐标上，u_orig, v_orig 可能不在图像范围内（例如负值或超过边界），需检查
            if 0 <= u_orig < w and 0 <= v_orig < h:
                # 双线性插值从 rectified_image 取值 (此处简单用最近邻插值示例)
                # 由于 u_orig, v_orig 为浮点数，这里使用 cv2.INTER_LINEAR 插值更合适
                # 我们先简单最近邻，当需高质量时可使用更复杂插值
                px_val = rectified_image[int(round(v_rect)), int(round(u_rect))]
                recovered[int(round(v_orig)), int(round(u_orig))] = px_val

    return recovered

#######################
# 示例流程
#######################
if __name__ == "__main__":
    # 假设你有一张校正后的左图像
    rectified_left = cv2.imread(r'G:\code\kiwirobot\rectified_images_20241206_150636/left_image_2560x1440_rectified.png')  # 用户需要提供

    # 使用给定参数逆向恢复
    # 注意：这里 dist_left、R1、P1、stereo_K_left、mono_K_left、mono_dist_left 必须与你最初矫正用的参数一致
    recovered_left = recover_original_image(rectified_left, R1, P1, stereo_K_left, dist_left, mono_K_left, mono_dist_left)

    cv2.imwrite(r"G:\code\kiwirobot\rectified_images_20241206_150636/Recovered Left.png", recovered_left)

