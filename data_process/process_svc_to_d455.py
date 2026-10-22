#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVC(KB 4参数) → 针孔 640x480 (SLAM/DA3 兼容版, 多进程)
- 顶部视野固定扩大 1.25 倍
- 左右视野固定扩大 1.2 倍
- 底部保持不变
- LANCZOS4 插值 (比 INTER_LINEAR 清晰)
- 不做任何锐化/对比度/亮度增强, 最大化 SLAM 和 DA3 兼容性
- 支持多进程加速

用法:
  python3 process_svc_to_d455.py --scene-dir /path/to/scene_1
  python3 process_svc_to_d455.py --data-dir /path/to/parent --workers 4
"""

import cv2
import numpy as np
import os
import glob
import argparse
import sys
import time
import multiprocessing as mp
import contextlib
import io


# ====================================================================
# 相机内参 (来自 extrinsic_svc_d455_0821.txt)
# ====================================================================

K_SVC = np.array([
    [413.4521,   0.0,     959.5152],
    [  0.0,    412.6878,  768.0989],
    [  0.0,      0.0,       1.0   ],
], dtype=np.float64)

DIST_SVC = np.array(
    [0.17030521, -0.00308843, -0.01711661, 0.00219803],
    dtype=np.float64
)

K_TARGET_BASE = np.array([
    [381.6182,   0.0,     320.0508],
    [  0.0,    381.2833,  241.8335],
    [  0.0,      0.0,       1.0   ],
], dtype=np.float64)

OUT_W, OUT_H = 640, 480

SVC_SIZE = (1920, 1536)

# ====================================================================
# 固定放大倍率
# ====================================================================
TOP_RATIO = 1.25    # 顶部视野扩大 1.25 倍
SIDE_RATIO = 1.2    # 左右视野扩大 1.2 倍


# ====================================================================
# 目标内参计算
# ====================================================================

def compute_target_K(top_ratio=TOP_RATIO, side_ratio=SIDE_RATIO):
    H = OUT_H - 1
    W = OUT_W - 1
    fy0 = K_TARGET_BASE[1, 1]
    fx0 = K_TARGET_BASE[0, 0]
    cx0 = K_TARGET_BASE[0, 2]
    cy0 = K_TARGET_BASE[1, 2]

    tan_top = cy0 / fy0
    tan_bot = (H - cy0) / fy0
    tan_left = cx0 / fx0
    tan_right = (W - cx0) / fx0

    tan_top_new = tan_top * top_ratio
    tan_bot_new = tan_bot
    tan_left_new = tan_left * side_ratio
    tan_right_new = tan_right * side_ratio

    fy_new = H / (tan_top_new + tan_bot_new)
    fx_new = W / (tan_left_new + tan_right_new)

    cy_new = fy_new * tan_top_new
    cx_new = cx0

    return np.array([
        [fx_new, 0.0,    cx_new],
        [0.0,    fy_new, cy_new],
        [0.0,    0.0,    1.0],
    ], dtype=np.float64)


# ====================================================================
# 映射表构建
# ====================================================================

def build_remap(new_K):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_SVC, DIST_SVC, np.eye(3), new_K,
        (OUT_W, OUT_H), cv2.CV_32FC1
    )
    return map1, map2


# ====================================================================
# 图像处理
# ====================================================================

def extract_timestamp_ms(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    return basename.split('_')[-1]


def process_single_image(input_path, output_path, map1, map2, svc_size):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")

    # 若输入尺寸与 SVC_SIZE 不一致, 底部补黑边对齐坐标系
    if img.shape[:2] != (svc_size[1], svc_size[0]):
        padded = np.zeros((svc_size[1], svc_size[0], 3), dtype=img.dtype)
        h = min(img.shape[0], svc_size[1])
        w = min(img.shape[1], svc_size[0])
        padded[:h, :w] = img[:h, :w]
        img = padded

    # LANCZOS4 插值 (比 INTER_LINEAR 保留更多高频细节)
    result = cv2.remap(img, map1, map2, cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv2.imwrite(output_path, result)


def _worker(args_tuple):
    input_path, output_path, map1, map2, svc_size = args_tuple
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            process_single_image(input_path, output_path, map1, map2, svc_size)
        return (os.path.basename(input_path), None)
    except Exception as e:
        return (os.path.basename(input_path), str(e))


# ====================================================================
# 主函数
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SVC(KB 4参数) → 针孔 640x480 (SLAM/DA3 兼容版, 多进程)'
    )
    parser.add_argument('--scene-dir', type=str, help='单个 scene 目录')
    parser.add_argument('--data-dir', type=str, help='包含多个 scene_N 的根目录')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行进程数 (默认 1)')
    parser.add_argument('--output-suffix', type=str, default='svc_processed',
                        help='输出子目录名 (默认: svc_processed)')
    parser.add_argument('--svc-size', type=str, default='1920x1536',
                        help='SVC 原始图像尺寸 (默认 1920x1536)')
    args = parser.parse_args()

    if not args.scene_dir and not args.data_dir:
        parser.print_help()
        sys.exit(1)

    w_str, h_str = args.svc_size.lower().split('x')
    svc_size_local = (int(w_str), int(h_str))

    print("=" * 60)
    print("SVC(KB 4参数) → 针孔 640x480 (SLAM/DA3 兼容版, 多进程)")
    print(f"顶部放大倍率: {TOP_RATIO}, 左右放大倍率: {SIDE_RATIO}")
    print(f"并行进程数: {args.workers}")
    print("=" * 60)
    print(f"SVC 内参:")
    print(f"  fx={K_SVC[0,0]:.2f}, fy={K_SVC[1,1]:.2f}, "
          f"cx={K_SVC[0,2]:.2f}, cy={K_SVC[1,2]:.2f}")
    print(f"  dist(KB 4) = {DIST_SVC.tolist()}")
    print(f"基准 D455 内参:")
    print(f"  fx={K_TARGET_BASE[0,0]:.2f}, fy={K_TARGET_BASE[1,1]:.2f}, "
          f"cx={K_TARGET_BASE[0,2]:.2f}, cy={K_TARGET_BASE[1,2]:.2f}")
    print(f"处理策略: LANCZOS4 插值, 无任何像素级增强")

    new_K = compute_target_K()

    print(f"\n新目标内参:")
    print(f"  fx={new_K[0,0]:.2f}, fy={new_K[1,1]:.2f}, "
          f"cx={new_K[0,2]:.2f}, cy={new_K[1,2]:.2f}")

    H = OUT_H - 1
    W = OUT_W - 1
    fov_top = np.rad2deg(np.arctan(new_K[1, 2] / new_K[1, 1]))
    fov_bot = np.rad2deg(np.arctan((H - new_K[1, 2]) / new_K[1, 1]))
    fov_left = np.rad2deg(np.arctan(new_K[0, 2] / new_K[0, 0]))
    fov_right = np.rad2deg(np.arctan((W - new_K[0, 2]) / new_K[0, 0]))
    print(f"\n实际 FOV:")
    print(f"  上方 {fov_top:.1f}°, 下方 {fov_bot:.1f}°")
    print(f"  左方 {fov_left:.1f}°, 右方 {fov_right:.1f}°")
    print(f"  垂直 FOV = {fov_top + fov_bot:.1f}°, "
          f"水平 FOV = {fov_left + fov_right:.1f}°")

    map1, map2 = build_remap(new_K)

    valid = (map1 >= 0) & (map1 < svc_size_local[0]) & \
            (map2 >= 0) & (map2 < svc_size_local[1])
    valid_ratio = np.sum(valid) / valid.size * 100
    print(f"\n有效像素: {valid_ratio:.1f}%")

    print("四角映射到 SVC 坐标:")
    corners = [((0, 0), 'TL'), ((OUT_W-1, 0), 'TR'),
               ((0, OUT_H-1), 'BL'), ((OUT_W-1, OUT_H-1), 'BR')]
    for (u, v), name in corners:
        us, vs = map1[v, u], map2[v, u]
        in_content = (0 <= us < svc_size_local[0]) and \
                     (0 <= vs < svc_size_local[1])
        status = '✓' if in_content else '✗'
        print(f"  {name}({u},{v}): → SVC({us:.1f},{vs:.1f}) {status}")

    # 收集 scene
    scenes = []
    if args.scene_dir:
        scenes.append(args.scene_dir)
    else:
        for d in sorted(glob.glob(os.path.join(args.data_dir, 'scene_*'))):
            if os.path.isdir(os.path.join(d, 'svc')):
                scenes.append(d)

    if not scenes:
        print("错误: 未找到包含 svc/ 子目录的场景")
        sys.exit(1)

    print(f"\n找到 {len(scenes)} 个场景待处理")

    # ---- 处理每个 scene ----
    for scene_dir in scenes:
        scene_name = os.path.basename(scene_dir)
        svc_dir = os.path.join(scene_dir, 'svc')
        output_dir = os.path.join(scene_dir, args.output_suffix)

        if not os.path.isdir(svc_dir):
            print(f"  [{scene_name}] 跳过: 无 svc/ 目录")
            continue

        jpg_files = sorted(glob.glob(os.path.join(svc_dir, '*.jpg')))
        if not jpg_files:
            print(f"  [{scene_name}] 跳过: 无 JPG 文件")
            continue

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n  [{scene_name}] 处理 {len(jpg_files)} 张 → {output_dir}/")

        t_start = time.time()

        if args.workers > 1:
            # 多进程
            tasks = [
                (p,
                 os.path.join(output_dir, f"{extract_timestamp_ms(p)}000000.png"),
                 map1, map2, svc_size_local)
                for p in jpg_files
            ]
            failed = []
            n_done = 0
            with mp.Pool(args.workers) as pool:
                for filename, err in pool.imap_unordered(_worker, tasks):
                    n_done += 1
                    if err:
                        failed.append((filename, err))
                    if n_done % 100 == 0 or n_done == len(tasks):
                        elapsed = time.time() - t_start
                        fps = n_done / elapsed if elapsed > 0 else 0
                        print(f"    [{n_done}/{len(tasks)}] "
                              f"失败: {len(failed)} "
                              f"({fps:.1f} fps)")
            for filename, err in failed:
                print(f"    错误: {filename}: {err}")
        else:
            # 单进程
            n_done = 0
            for jpg_path in jpg_files:
                ts = extract_timestamp_ms(jpg_path)
                out_path = os.path.join(output_dir, f"{ts}000000.png")
                try:
                    process_single_image(jpg_path, out_path, map1, map2,
                                         svc_size_local)
                    n_done += 1
                    if n_done % 100 == 0:
                        elapsed = time.time() - t_start
                        fps = n_done / elapsed if elapsed > 0 else 0
                        print(f"    [{n_done}/{len(jpg_files)}] "
                              f"已耗时 {elapsed:.1f}s ({fps:.1f} fps)")
                except Exception as e:
                    print(f"    错误: {os.path.basename(jpg_path)}: {e}")

        elapsed = time.time() - t_start
        avg_fps = n_done / elapsed if elapsed > 0 else 0
        print(f"    [{n_done}/{len(jpg_files)}] 完成")
        print(f"  总耗时: {elapsed:.2f}s, 平均 {avg_fps:.1f} fps")

    print("\n全部处理完成!")


if __name__ == '__main__':
    # Linux 上使用 'fork' 启动方式更快 (默认)
    # Windows 上需要 'spawn'
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    main()