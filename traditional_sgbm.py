#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo Matching (SGBM) - No ximgproc version
- Preprocess (CLAHE + Gaussian)
- SGBM disparity (left & right)
- Left-Right Consistency Check (LRC)
- Speckle removal & median smoothing
"""

import cv2
import numpy as np
import argparse
import os

def ensure_multiple_of_16(x):
    return int(np.ceil(x/16.0)*16)

def preprocess(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

def left_right_consistency_check(dispL, dispR, max_diff=1):
    h, w = dispL.shape
    dispL_f = dispL.astype(np.float32) / 16.0
    dispR_f = dispR.astype(np.float32) / 16.0

    xs = np.arange(w)[None, :].repeat(h, axis=0)
    xr = (xs - dispL_f).round().astype(np.int32)
    xr = np.clip(xr, 0, w-1)

    diff = np.abs(dispL_f - dispR_f[np.arange(h)[:,None], xr])
    mask = (diff <= max_diff) & (dispL > 0)
    disp_checked = dispL.copy()
    disp_checked[~mask] = 0
    return disp_checked

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default=r"C:\Users\jingx\Desktop\img_test\left/1.jpeg", help="Left image (8-bit)")
    parser.add_argument("--right", default=r"C:\Users\jingx\Desktop\img_test\right/1.jpeg", help="Right image (8-bit)")
    parser.add_argument("--out_disp", default="disp.png")
    parser.add_argument("--out_viz",  default="disp_viz.png")
    parser.add_argument("--numDisparities", type=int, default=24)
    parser.add_argument("--blockSize", type=int, default=3)
    parser.add_argument("--uniquenessRatio", type=int, default=10)
    parser.add_argument("--speckleWindowSize", type=int, default=100)
    parser.add_argument("--speckleRange", type=int, default=2)
    parser.add_argument("--disp12MaxDiff", type=int, default=1)
    args = parser.parse_args()

    args.numDisparities = ensure_multiple_of_16(max(16, args.numDisparities))
    if args.blockSize % 2 == 0:
        args.blockSize += 1

    left  = cv2.imread(args.left,  cv2.IMREAD_COLOR)
    right = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError("Cannot load input images.")

    gl = preprocess(left)
    gr = preprocess(right)

    # 左匹配器
    matcherL = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=args.numDisparities,
        blockSize=args.blockSize,
        P1=8*args.blockSize*args.blockSize,
        P2=32*args.blockSize*args.blockSize,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        uniquenessRatio=args.uniquenessRatio,
        speckleWindowSize=args.speckleWindowSize,
        speckleRange=args.speckleRange,
        disp12MaxDiff=args.disp12MaxDiff
    )
    # 右匹配器
    matcherR = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=args.numDisparities,
        blockSize=args.blockSize,
        P1=8*args.blockSize*args.blockSize,
        P2=32*args.blockSize*args.blockSize,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        uniquenessRatio=args.uniquenessRatio,
        speckleWindowSize=args.speckleWindowSize,
        speckleRange=args.speckleRange,
        disp12MaxDiff=args.disp12MaxDiff
    )

    dispL = matcherL.compute(gl, gr)
    dispR = matcherR.compute(gr, gl)

    # 一致性检查
    disp_checked = left_right_consistency_check(dispL, dispR, args.disp12MaxDiff)

    # speckle filtering
    disp16 = disp_checked.astype(np.int16)
    cv2.filterSpeckles(disp16, 0, 200, 32)

    # median smoothing
    disp_float = disp16.astype(np.float32)
    disp_float = cv2.medianBlur(disp_float, 3)

    # 保存16位视差图
    cv2.imwrite(args.out_disp, np.clip(disp_float, 0, None).astype(np.uint16))

    # 保存可视化图
    disp_vis = cv2.normalize(disp_float/16.0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(args.out_viz, disp_vis)

    print(f"[OK] disparity saved: {args.out_disp}")
    print(f"[OK] visualization saved: {args.out_viz}")

if __name__ == "__main__":
    main()
