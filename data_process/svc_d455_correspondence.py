#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVC处理后图像 ←→ D455深度图 像素精确对齐

将 D455 深度图精确重投影到 SVC 处理后图像的坐标系中，
利用已知深度值和标定外参消除平移视差，使两个图像可以逐像素对比评测。

用法:
  # 单张对齐
  python3 svc_d455_correspondence.py --depth /path/to/depth.png --out /path/to/warped_depth.png

  # 批量对齐 (scene目录下所有深度图)
  python3 svc_d455_correspondence.py --scene-dir /path/to/scene_1 --depth-dir depth_processed

  # 打印参数信息
  python3 svc_d455_correspondence.py --print
"""

import cv2
import numpy as np
import argparse
import os
import sys
import glob
import time
import multiprocessing as mp
import contextlib
import io

# ====================================================================
# 内参
# ====================================================================

K_SVC_TARGET = np.array([
    [318.0152, 0.0,      320.0508],
    [0.0,      338.5520, 268.4133],
    [0.0,      0.0,      1.0],
], dtype=np.float64)

K_D455_RGB = np.array([
    [381.6182, 0.0,      320.0508],
    [0.0,      381.2833, 241.8335],
    [0.0,      0.0,      1.0],
], dtype=np.float64)

# ====================================================================
# 外参 (P_d455 = R @ P_svc + t)
# ====================================================================

R = np.array([
    [0.9988213,  -0.01314425, -0.04672521],
    [0.01744383,  0.99552857,  0.09283629],
    [0.04529602, -0.09354193,  0.99458443],
], dtype=np.float64)

t = np.array([-0.337857, -0.048525, 0.163150], dtype=np.float64)

OUT_W, OUT_H = 640, 480


# ====================================================================
# 距离近似 (仅旋转, 无深度)
# ====================================================================

def build_distant_remap():
    """构建远距离近似remap: D455像素 → SVC像素 (忽略平移)"""
    K_d_inv = np.linalg.inv(K_D455_RGB)

    yv, xv = np.meshgrid(np.arange(OUT_H), np.arange(OUT_W), indexing='ij')
    ones = np.ones_like(xv, dtype=np.float64)
    pixels = np.stack([xv, yv, ones], axis=-1)  # (H, W, 3)

    rays = pixels @ K_d_inv.T                    # D455射线
    rays_svc = rays @ R                          # 旋转到SVC坐标系 (R^T的逆 = R)
    pixels_svc = rays_svc @ K_SVC_TARGET.T
    u_s = pixels_svc[:, :, 0] / pixels_svc[:, :, 2]
    v_s = pixels_svc[:, :, 1] / pixels_svc[:, :, 2]

    return u_s.astype(np.float32), v_s.astype(np.float32)


def warp_depth_distant(depth_img):
    """远距离近似: 用旋转重映射深度图到SVC坐标系"""
    u_s, v_s = build_distant_remap()
    return cv2.remap(depth_img, u_s, v_s, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ====================================================================
# 精确模式 (利用深度值 + 外参平移)
# ====================================================================

def warp_depth_exact(depth_img):
    """
    利用深度值精确重投影: D455深度图 → SVC坐标系。

    对每个D455像素(u_d, v_d)有深度z_d:
      P_d = z_d * K_d455^{-1} * [u_d, v_d, 1]   (D455坐标系3D点)
      P_s = R^T @ (P_d - t)                       (SVC坐标系3D点)
      [u_s, v_s, 1] ~ K_svc @ P_s                (SVC像素)

    多个D455像素可能映射到同一SVC像素，取最小深度(最近点)。

    返回: (warped_depth, valid_mask, coverage)
    """
    depth = depth_img.astype(np.float64)
    valid_d = depth > 0

    if np.sum(valid_d) == 0:
        return np.zeros((OUT_H, OUT_W), dtype=np.uint16), np.zeros((OUT_H, OUT_W), dtype=bool), 0.0

    # 只处理有效深度像素
    yv, xv = np.where(valid_d)
    z_vals = depth[valid_d]

    # 1. D455像素 → 射线方向
    K_d_inv = np.linalg.inv(K_D455_RGB)
    rays_d = np.stack([xv.astype(np.float64), yv.astype(np.float64),
                        np.ones(len(xv), dtype=np.float64)], axis=1)  # (N, 3)
    rays_d = rays_d @ K_d_inv.T  # (N, 3)

    # 2. 3D点 (D455坐标系)
    P_d = rays_d * z_vals[:, np.newaxis]  # (N, 3)

    # 3. 转换到SVC坐标系: P_s = R^T @ (P_d - t)
    R_T = R.T
    P_s = (P_d - t[np.newaxis, :]) @ R_T  # (N, 3)

    # 4. 投影到SVC像素
    P_s_proj = P_s @ K_SVC_TARGET.T  # (N, 3)
    u_s = P_s_proj[:, 0] / P_s_proj[:, 2]
    v_s = P_s_proj[:, 1] / P_s_proj[:, 2]

    # 过滤越界
    valid_s = (u_s >= 0) & (u_s < OUT_W) & (v_s >= 0) & (v_s < OUT_H)
    u_s = u_s[valid_s]
    v_s = v_s[valid_s]
    z_vals = z_vals[valid_s]
    P_s_z = P_s[valid_s, 2]  # SVC坐标系下的深度

    if len(u_s) == 0:
        return np.zeros((OUT_H, OUT_W), dtype=np.uint16), np.zeros((OUT_H, OUT_W), dtype=bool), 0.0

    # 5. 取整填充到SVC坐标系深度图
    # 多个点映射到同一像素时取最近(z最小)
    u_i = np.round(u_s).astype(np.int32)
    v_i = np.round(v_s).astype(np.int32)

    warped = np.full((OUT_H, OUT_W), np.inf, dtype=np.float64)
    # 向量化z-buffer
    for i in range(len(u_i)):
        if P_s_z[i] < warped[v_i[i], u_i[i]]:
            warped[v_i[i], u_i[i]] = z_vals[i]

    warped = np.where(np.isinf(warped), 0, warped).astype(np.uint16)
    mask = warped > 0

    return warped, mask, np.sum(mask) / (OUT_W * OUT_H) * 100


# ====================================================================
# 单文件处理
# ====================================================================

def process_single(depth_path, output_path, mode='exact'):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"无法读取: {depth_path}")

    if mode == 'exact':
        warped, mask, coverage = warp_depth_exact(depth)
    else:
        warped = warp_depth_distant(depth)
        mask = warped > 0
        coverage = np.sum(mask) / mask.size * 100

    cv2.imwrite(output_path, warped)

    # 同时输出有效像素mask
    mask_path = output_path.replace('.png', '_mask.png')
    cv2.imwrite(mask_path, (mask.astype(np.uint8) * 255))

    return coverage


def _worker(args_tuple):
    depth_path, output_path, mode = args_tuple
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coverage = process_single(depth_path, output_path, mode)
        return (os.path.basename(depth_path), coverage, None)
    except Exception as e:
        return (os.path.basename(depth_path), 0.0, str(e))


# ====================================================================
# 主函数
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='D455深度图 → SVC坐标系精确重投影')
    parser.add_argument('--depth', type=str, default=None,
                        help='单张D455深度图路径')
    parser.add_argument('--out', type=str, default=None,
                        help='输出路径 (默认: 输入路径加_warped后缀)')
    parser.add_argument('--scene-dir', type=str, default=None,
                        help='scene目录 (批量模式, 配合--depth-dir)')
    parser.add_argument('--depth-dir', type=str, default='depth_processed',
                        help='scene下深度图子目录名 (默认: depth_processed)')
    parser.add_argument('--out-dir', type=str, default='depth_svc_aligned',
                        help='输出子目录名 (默认: depth_svc_aligned)')
    parser.add_argument('--mode', type=str, choices=['exact', 'distant'],
                        default='exact', help='重投影模式')
    parser.add_argument('--workers', type=int, default=1, help='并行进程数')
    parser.add_argument('--print', action='store_true', default=False,
                        help='打印参数信息')
    args = parser.parse_args()

    if args.print or (not args.depth and not args.scene_dir):
        print_info(args.mode)

    if args.depth:
        out = args.out or args.depth.replace('.png', '_warped.png')
        coverage = process_single(args.depth, out, args.mode)
        print(f"完成: {out}")
        print(f"有效像素覆盖率: {coverage:.1f}%")

    elif args.scene_dir:
        depth_dir = os.path.join(args.scene_dir, args.depth_dir)
        if not os.path.isdir(depth_dir):
            print(f"错误: 深度图目录不存在: {depth_dir}")
            sys.exit(1)

        png_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        if not png_files:
            print(f"错误: {depth_dir} 中无PNG文件")
            sys.exit(1)

        out_dir = os.path.join(args.scene_dir, args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"模式: {args.mode}, {len(png_files)} 张深度图 → {out_dir}/")

        tasks = [(p, os.path.join(out_dir, os.path.basename(p)), args.mode)
                 for p in png_files]
        t_start = time.time()

        if args.workers > 1:
            failed, n_done = [], 0
            with mp.Pool(args.workers) as pool:
                for fname, cov, err in pool.imap_unordered(_worker, tasks):
                    n_done += 1
                    if err:
                        failed.append((fname, err))
                    if n_done % 100 == 0 or n_done == len(tasks):
                        print(f"  [{n_done}/{len(tasks)}] 失败: {len(failed)}")
            for f, e in failed:
                print(f"  错误: {f}: {e}")
        else:
            n_done = 0
            for depth_path, out_path, mode in tasks:
                coverage = process_single(depth_path, out_path, mode)
                n_done += 1
                if n_done % 100 == 0:
                    print(f"  [{n_done}/{len(tasks)}]")

        elapsed = time.time() - t_start
        print(f"完成: {n_done}张, 耗时 {elapsed:.1f}s")

    print("\n全部完成!")


def print_info(mode):
    print("=" * 60)
    print(f"SVC处理后图像 ←→ D455深度图 像素对齐 (模式: {mode})")
    print("=" * 60)
    print(f"\nSVC目标K:  fx={K_SVC_TARGET[0,0]:.2f}, fy={K_SVC_TARGET[1,1]:.2f}, "
          f"cx={K_SVC_TARGET[0,2]:.2f}, cy={K_SVC_TARGET[1,2]:.2f}")
    print(f"D455 K:    fx={K_D455_RGB[0,0]:.2f}, fy={K_D455_RGB[1,1]:.2f}, "
          f"cx={K_D455_RGB[0,2]:.2f}, cy={K_D455_RGB[1,2]:.2f}")

    u_s, v_s = build_distant_remap()
    print(f"\n外参: t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")

    print(f"\n--- D455深度图 → SVC图像 (忽略平移) ---")
    for (u, v), name in [((0, 0), 'TL'), ((639, 0), 'TR'),
                          ((0, 479), 'BL'), ((639, 479), 'BR')]:
        valid_svc = 0 <= u_s[v, u] < OUT_W and 0 <= v_s[v, u] < OUT_H
        print(f"  D455 {name}({u},{v}) → SVC ({u_s[v,u]:.1f}, {v_s[v,u]:.1f}) "
              f"{'✓' if valid_svc else '✗'}")

    # 覆盖统计
    valid_mask = (u_s >= 0) & (u_s < OUT_W) & (v_s >= 0) & (v_s < OUT_H)
    print(f"\nD455在SVC中的覆盖: {np.sum(valid_mask)}/{OUT_W*OUT_H} "
          f"({np.sum(valid_mask)/(OUT_W*OUT_H)*100:.1f}%)")

    if mode == 'exact':
        print("\n精确模式说明:")
        print("  利用D455深度值 + 外参(R,t)消除平移视差")
        print("  对每个D455像素: 3D→旋转平移→重新投影→SVC像素")
        print("  多个点映射到同一像素时取最近点(z-buffer)")
        print("  输出图与SVC处理后的图像逐像素对应，可直接评测")
    else:
        print("\n远距离近似说明:")
        print("  仅用旋转，忽略平移")
        print("  近处物体(<10m)有视差偏移")


if __name__ == '__main__':
    main()