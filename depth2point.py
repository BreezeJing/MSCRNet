import cv2
import numpy as np
import open3d as o3d

# 1. 读取深度图与彩色图
depth = cv2.imread(r"E:\Code\Stereo\20250630_220756/depth_20250630_220756.png", cv2.IMREAD_UNCHANGED)  # shape: (H_d, W_d)
depth = depth/10
color = cv2.imread(r"E:\Code\Stereo\20250630_220756\left_20250630_220756.png", cv2.IMREAD_COLOR)      # shape: (H_c, W_c, 3)
scale_img = 2
H_d, W_d = depth.shape
# 2. 将彩色图下采样到深度分辨率
#    使用 INTER_AREA 插值，对图像内容缩小效果更好
color_ds = cv2.resize(color, (W_d, H_d), interpolation=cv2.INTER_LINEAR)

# 3. 深度相机内参（需替换为实际标定值）
fx, fy = 1087.47099650264/scale_img, 1087.69801067851/scale_img       # 深度相机焦距（像素）
cx, cy = 958.792331728028/scale_img, 520.784905211636/scale_img  # 主点坐标
K_d = np.array([[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]], dtype=np.float64)

# 4. 反投影生成点云
scale = 1000.0  # 若深度单位为毫米，否则设为1
pts = []
cols = []

for v in range(H_d):
    for u in range(W_d):
        z = float(depth[v, u]) / scale
        if z <= 0: continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts.append((x, y, z))
        b, g, r = color_ds[v, u]
        cols.append((r/255.0, g/255.0, b/255.0))

# 5. 构建 Open3D 点云并保存
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float32))
pcd.colors = o3d.utility.Vector3dVector(np.array(cols, dtype=np.float32))
o3d.io.write_point_cloud("downsampled_color_pointcloud.ply", pcd)
print("点云已保存到 downsampled_color_pointcloud.ply")
