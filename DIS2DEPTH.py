import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='批量处理视差图，生成深度图，并可选择性地进行深度补全。')
    parser.add_argument('--disparity_dir', type=str, default=r'G:\data\LYL_20240915_kiwi\JR-1\Organized data\output_depth', help='视差图所在目录')
    parser.add_argument('--rgb_dir', type=str, help='RGB图像所在目录，仅在启用深度补全时需要')
    parser.add_argument('--output_dir', type=str, default=r'G:\data\LYL_20240915_kiwi\JR-1\Organized data\depth/', help='输出深度图的目录')
    parser.add_argument('--enable_completion', action='store_true', default=False, help='启用深度补全')
    parser.add_argument('--focal_length', type=float, default=1010/2, help='相机焦距（像素）')
    parser.add_argument('--baseline', type=float, default=25*256, help='相机基线距离（米）')
    parser.add_argument('--min_disparity', type=float, default=0.1, help='最小有效视差')
    parser.add_argument('--max_percentile', type=float, default=99.5, help='最大视差的百分位数，用于过滤异常值')
    parser.add_argument('--diameter', type=int, default=9, help='引导双边滤波的直径')
    parser.add_argument('--sigma_color', type=float, default=75, help='引导双边滤波的色彩标准差')
    parser.add_argument('--sigma_space', type=float, default=75, help='引导双边滤波的空间标准差')
    return parser.parse_args()


def get_matching_files(disparity_dir, rgb_dir, enable_completion):
    disparity_files = sorted([f for f in os.listdir(disparity_dir) if os.path.isfile(os.path.join(disparity_dir, f))])

    if enable_completion:
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))])
        matched_files = []
        for disp_file in disparity_files:
            # 提取编号，例如 disp_01.png -> 01
            identifier = os.path.splitext(disp_file)[0].split('_')[-1]
            # 假设RGB图像以 img_开头
            rgb_file = f'img_{identifier}.png'
            if rgb_file in rgb_files:
                matched_files.append((disp_file, rgb_file))
            else:
                print(f"警告：找不到对应的RGB图像 for {disp_file}")
        return matched_files
    else:
        # 如果未启用补全，返回仅包含视差图的文件列表
        return [(disp_file, None) for disp_file in disparity_files]


def filter_disparity(disparity_map, min_disp, max_disp):
    valid_mask = (disparity_map > min_disp) & (disparity_map < max_disp)
    filtered_disparity = np.where(valid_mask, disparity_map, 0).astype(np.float32)
    return filtered_disparity


def convert_disparity_to_depth(filtered_disparity, focal_length, baseline):
    # 避免除以零，将视差为0的设置为一个较大的深度值（例如10米）
    filtered_disparity[filtered_disparity == 0] = 0.1  # 或者一个非常小的非零值
    depth_map = (focal_length * baseline) / filtered_disparity  # 单位：米
    depth_map_mm = (depth_map).astype(np.uint16)  # 单位：毫米
    return depth_map, depth_map_mm


def complete_depth(depth_map, rgb_image, diameter, sigma_color, sigma_space):
    try:
        joint_bilateral_filter = cv2.ximgproc.jointBilateralFilter
    except AttributeError:
        raise ImportError("未找到ximgproc模块，请安装opencv-contrib-python。")

    depth_map_float = depth_map.copy()
    completed_depth_float = joint_bilateral_filter(
        guide=rgb_image,
        src=depth_map_float,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    completed_depth_mm = (completed_depth_float).astype(np.uint16)
    return completed_depth_mm


def process_image_pair(disp_path, rgb_path, output_path, args):
    # 读取视差图（假设为单通道16位图像）
    disparity_map = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    if disparity_map is None:
        print(f"错误：无法加载视差图 {disp_path}")
        return

    # 第一步：过滤异常的视差值
    valid_disparity_values = disparity_map[disparity_map > 0]
    if len(valid_disparity_values) == 0:
        print(f"警告：视差图中没有有效的视差值 {disp_path}")
        return
    max_disparity = np.percentile(valid_disparity_values, args.max_percentile)
    filtered_disparity = filter_disparity(disparity_map, args.min_disparity, max_disparity)

    # 第二步：将视差图转换为深度图
    depth_map, depth_map_mm = convert_disparity_to_depth(filtered_disparity, args.focal_length, args.baseline)

    # 第三步（可选）：结合RGB图像对深度图进行补全
    if args.enable_completion:
        if rgb_path is None:
            print(f"错误：未提供 RGB 图像，无法进行深度补全 {disp_path}")
            return

        # 读取RGB图像
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            print(f"错误：无法加载RGB图像 {rgb_path}")
            return

        # 确保RGB图像和深度图尺寸一致
        if rgb_image.shape[:2] != depth_map_mm.shape[:2]:
            rgb_image = cv2.resize(rgb_image, (depth_map_mm.shape[1], depth_map_mm.shape[0]))

        try:
            completed_depth_mm = complete_depth(depth_map, rgb_image, args.diameter, args.sigma_color, args.sigma_space)
            final_depth_mm = completed_depth_mm
        except ImportError as e:
            print(f"错误：{e}")
            return
    else:
        # 如果未启用深度补全，直接使用原始深度图
        final_depth_mm = depth_map_mm

    # 保存16位深度图
    cv2.imwrite(output_path, final_depth_mm)


def main():
    args = parse_arguments()

    # 如果启用了深度补全，则必须提供RGB目录
    if args.enable_completion and args.rgb_dir is None:
        print("错误：启用深度补全时必须提供RGB图像目录。")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取匹配的文件对
    matched_files = get_matching_files(args.disparity_dir, args.rgb_dir, args.enable_completion)
    if not matched_files:
        print("错误：没有找到匹配的视差图和RGB图像文件。")
        return

    print(f"找到 {len(matched_files)} 对匹配的视差图和RGB图像。")

    # 批量处理
    for disp_file, rgb_file in tqdm(matched_files, desc="处理图像对"):
        disp_path = os.path.join(args.disparity_dir, disp_file)
        output_filename = f'completed_depth_{os.path.splitext(disp_file)[0]}.png'
        output_path = os.path.join(args.output_dir, output_filename)
        process_image_pair(disp_path, rgb_file, output_path, args)

    print("所有图像处理完成。")


if __name__ == "__main__":
    main()
