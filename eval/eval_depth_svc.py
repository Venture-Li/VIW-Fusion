#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVC 深度评测: VINS 点云深度 vs D455 深度真值 (严格全外参版)

与 eval_depth_4m_wheel.py (D455 版) 的区别:
  D455 版中 VINS 相机与深度相机同光心, est/gt 都是 D455 系 z 值, 直接同像素对比。
  SVC 与 D455 光心不重合 (平移 t≈0.15m), 存在随距离变化的视差,
  本脚本用全外参链消除:
    1) VINS 点 (世界系) → IMU → D455 相机系, 投影到深度图像素 (K_depth, 无畸变)
    2) 取该像素深度真值 z_gt, 反投影为 3D 点 (D455 光心)
    3) P_svc = R^T (P_d455 - t)  变换到 SVC 光心, 全外参 (R+t) 处理视差
    4) est_depth = VINS 点在 SVC 系的 z; gt_depth = 真值点在 SVC 系的 z
  二者是"同一表面点从 SVC 光心观测的 z 深度", 严格可比。

坐标系约定 (extrinsic_svc_d455_1018_fused.txt): P_d455 = R @ P_svc + t
处理后的 SVC 针孔图 (svc_processed) 内参 = K_depth, 光轴已对齐 D455。
"""

import numpy as np
import cv2
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 用户配置（仅修改这里！） ====================
data_root = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/data/scene_1/"
pose_file = "/home/nio/vscode/python_project_backup/VIW-Fusion/output/pose_output.txt"
points_file = "/home/nio/vscode/python_project_backup/VIW-Fusion/output/points_output.txt"
frame_timestamp = 1792312506.500302   # 秒 (VINS 位姿时间戳)

# 图像目录
cam_subdir = "svc_processed"          # 处理后的 SVC 针孔图 (640x480, 对齐深度图)
depth_subdir = "depth0"               # D455 原始深度图 (无畸变, 直接用)

# D455 深度相机内参 (scene_N/intrinsics.txt Depth 段, 畸变为0)
# 处理后的 SVC 针孔图使用同一内参, 两侧投影共用
K_D = np.array([
    [386.7249, 0,        323.3481],
    [0,        386.7249, 233.6760],
    [0,        0,        1       ]
])

# SVC -> D455 外参: P_d455 = R_sd @ P_svc + t_sd
R_sd = np.array([
    [ 0.99176726,  0.03108473, -0.12422338],
    [-0.03676192,  0.99836928, -0.04367313],
    [ 0.12266324,  0.04788027,  0.99129270]
])
t_sd = np.array([-0.021910, 0.123421, 0.096283])

# D455 IMU → cam0 外参 (cam0.yaml, body_T_cam0: P_imu = R_bc @ P_cam + t_bc)
R_bc = np.array([
    [ 0.999994,  0.000078,  0.003535],
    [-0.000097,  0.999987,  0.005146],
    [-0.003535, -0.005147,  0.999981]
])
t_bc = np.array([-0.028933, -0.007453, -0.015558])
R_ic = R_bc.T               # P_cam = R_ic @ (P_imu - t_bc) 中的旋转
t_ic = -R_ic @ t_bc

DEPTH_MAX = 3.0             # 评测最大深度 (米), 与 D455 版一致

# 输出文件
output_image = "./projected_points_frame.png"
overlay_image = "./color_depth_overlay.png"
depth_scatter = "./depth_scatter.png"
comparison_csv = "./depth_comparison.csv"
# =================================================================


def parse_pose_line(line):
    parts = line.strip().split()
    t = float(parts[0])
    p = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    return t, p, q


def quat_to_rot(q):
    qx, qy, qz, qw = q
    return np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw  ],
        [2*qx*qy+2*qz*qw,   1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw  ],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx**2-2*qy**2]
    ])


def load_points_with_visibility(points_file):
    points, obs = [], []
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 3:
                continue
            points.append(parts[:3])
            obs.append(parts[3:] if len(parts) > 3 else [])
    return np.array(points), obs


def is_point_visible(target_stamp, obs_list, tol=1e-4):
    for s in obs_list:
        if abs(s - target_stamp) < tol:
            return True
    return False


def find_cam_image(cam_dir, frame_timestamp):
    """cam_subdir 按 data.csv (cam0) 或文件名毫秒时间戳 (svc_processed) 找最近帧"""
    if os.path.isfile(os.path.join(cam_dir, "data.csv")):
        stamps, names = [], []
        with open(os.path.join(cam_dir, "data.csv")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                try:
                    stamps.append(int(parts[0]) * 1e-9)
                    names.append(parts[1])
                except ValueError:
                    continue
    else:
        stamps, names = [], []
        for p in glob.glob(os.path.join(cam_dir, "data", "*.png")):
            base = os.path.splitext(os.path.basename(p))[0]
            # svc_processed 文件名: <毫秒>000000.png
            ms = int(base) // 10**6
            stamps.append(ms * 1e-3)
            names.append(os.path.basename(p))
    i = int(np.argmin(np.abs(np.array(stamps) - frame_timestamp)))
    return names[i], stamps[i]


# ------------------- 1. 读取目标帧IMU位姿 -------------------
target_pose = None
with open(pose_file, 'r') as f:
    for line in f:
        t, p_wi, q_wi = parse_pose_line(line)
        if abs(t - frame_timestamp) < 1e-4:
            target_pose = (p_wi, q_wi)
            break
if target_pose is None:
    raise ValueError(f"未找到时间戳 {frame_timestamp}")

p_wi, q_wi = target_pose
R_wi = quat_to_rot(q_wi)
print(f"IMU位置 (世界系): {p_wi}")

# ------------------- 2. 读取处理后的 SVC 图 -------------------
cam_dir = os.path.join(data_root, cam_subdir)
cam_filename, cam_ts = find_cam_image(cam_dir, frame_timestamp)
cam_img_path = os.path.join(cam_dir, cam_filename)
color_img = cv2.imread(cam_img_path)
if color_img is None:
    raise FileNotFoundError(f"无法读取 SVC 图: {cam_img_path}")
h, w = color_img.shape[:2]
print(f"SVC 图: {cam_filename} (ts={cam_ts:.6f}s, 与位姿差 {abs(cam_ts-frame_timestamp)*1e3:.1f}ms)")
assert (h, w) == (480, 640), f"SVC 图尺寸异常: {w}x{h}"

# ------------------- 3. 读取深度图 -------------------
depth_csv = os.path.join(data_root, depth_subdir, "data.csv")
stamps, names = [], []
with open(depth_csv) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        try:
            stamps.append(int(parts[0]) * 1e-9)
            names.append(parts[1])
        except ValueError:
            continue
i = int(np.argmin(np.abs(np.array(stamps) - frame_timestamp)))
depth_img_path = os.path.join(data_root, depth_subdir, "data", names[i])
print(f"深度图: {names[i]} (ts={stamps[i]:.6f}s, 与位姿差 {abs(stamps[i]-frame_timestamp)*1e3:.1f}ms)")
depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
if depth_img is None:
    raise FileNotFoundError(f"无法读取深度图: {depth_img_path}")
depth_m = depth_img.astype(np.float32) / 1000.0   # 毫米转米

# ------------------- 4. 加载点云 -------------------
points, obs_list = load_points_with_visibility(points_file)
print(f"总点数: {len(points)}")
if len(points) == 0:
    raise ValueError("点云文件为空！")

# ------------------- 5. 投影与深度比较 (全外参, 处理视差) -------------------
fx, fy, cx, cy = K_D[0, 0], K_D[1, 1], K_D[0, 2], K_D[1, 2]

all_pixels = []          # 投影到 SVC 图内的点 (u_s, v_s)
valid_points_info = []   # (u_s, v_s, est_depth, gt_depth)

for i, P_w in enumerate(points):
    if obs_list[i] and not is_point_visible(frame_timestamp, obs_list[i]):
        continue

    # 世界 → 车体系(IMU) → svc_processed 相机系 (处理图坐标系 = D455 轴向, SVC 光心)
    P_i = R_wi.T @ (P_w - p_wi)
    P_p = R_c.T @ (P_i - t_c)         # VINS 点在处理图相机系
    if P_p[2] <= 0 or P_p[2] > DEPTH_MAX:
        continue

    # 处理图相机系 → D455 相机系 (P_d455 = R_sd @ P_p + t_sd), 投影取深度真值
    P_dc = R_sd @ P_p + t_sd
    u_d = fx * P_dc[0] / P_dc[2] + cx
    v_d = fy * P_dc[1] / P_dc[2] + cy
    u_di, v_di = int(round(u_d)), int(round(v_d))
    if not (0 <= u_di < w and 0 <= v_di < h):
        continue

    z_gt_d = depth_m[v_di, u_di]      # D455 测得的深度 (D455 光心)
    if z_gt_d <= 0 or z_gt_d > DEPTH_MAX:
        continue

    # 深度真值反投影为 3D 点 (D455 系), 变换回处理图相机系 (SVC 光心, D455 轴向)
    P_gt_d = np.array([(u_d - cx) / fx * z_gt_d,
                       (v_d - cy) / fy * z_gt_d,
                       z_gt_d])
    P_gt_s = R_sd.T @ (P_gt_d - t_sd)
    if P_gt_s[2] <= 0:
        continue
    gt_depth = P_gt_s[2]              # 真值点在 SVC 系的 z 深度

    # VINS 点在处理图相机系的 z 深度
    est_depth = P_p[2]

    # 投影到 SVC 针孔图 (内参 = K_D, 光轴已对齐)
    u_s = fx * P_p[0] / est_depth + cx
    v_s = fy * P_p[1] / est_depth + cy
    u_si, v_si = int(round(u_s)), int(round(v_s))
    if not (0 <= u_si < w and 0 <= v_si < h):
        continue

    all_pixels.append((u_si, v_si))
    valid_points_info.append((u_si, v_si, est_depth, gt_depth))

print(f"\n总投影点（SVC 图内）: {len(all_pixels)}")
print(f"有效深度点: {len(valid_points_info)}")

# ------------------- 6. 绘制投影点 -------------------
img_draw = color_img.copy()
for (u, v, d_est, d_gt) in valid_points_info:
    cv2.circle(img_draw, (u, v), 3, (0, 0, 255), -1)
cv2.putText(img_draw, f"Valid: {len(valid_points_info)}/{len(points)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite(output_image, img_draw)
print(f"投影图像已保存至: {output_image}")

# ------------------- 7. SVC 图 + 深度叠加图 -------------------
depth_colormap = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_m, alpha=255.0 / DEPTH_MAX), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(color_img, 0.6, depth_colormap, 0.4, 0)
cv2.imwrite(overlay_image, overlay)
print(f"SVC-深度叠加图已保存至: {overlay_image}")

# ------------------- 8. 像素分布图 -------------------
plt.figure(figsize=(8, 6))
if all_pixels:
    plt.scatter([p[0] for p in all_pixels], [p[1] for p in all_pixels],
                s=1, alpha=0.3, label='all projected')
if valid_points_info:
    plt.scatter([p[0] for p in valid_points_info], [p[1] for p in valid_points_info],
                s=2, c='red', label='valid depth')
plt.xlim(0, w); plt.ylim(h, 0)
plt.xlabel('u'); plt.ylabel('v')
plt.legend(); plt.title('Projected Points Distribution (SVC)')
plt.savefig('pixel_distribution_detailed.png'); plt.close()
print("详细像素分布图已保存至 pixel_distribution_detailed.png")

# ------------------- 9. 深度误差统计 -------------------
if valid_points_info:
    est_depths = np.array([d[2] for d in valid_points_info])
    gt_depths = np.array([d[3] for d in valid_points_info])
    slope = np.linalg.lstsq(gt_depths.reshape(-1, 1), est_depths, rcond=None)[0][0]
    print(f"\n深度回归斜率 (est/gt) = {slope:.4f}")

    plt.figure()
    plt.scatter(gt_depths, est_depths, alpha=0.6)
    plt.plot([0, max(gt_depths)], [0, slope * max(gt_depths)], 'r--',
             label=f'slope={slope:.2f}')
    plt.plot([0, max(gt_depths)], [0, max(gt_depths)], 'g--', label='ideal (slope=1)')
    plt.xlabel('GT Depth from SVC (m)')
    plt.ylabel('VINS Est Depth from SVC (m)')
    plt.legend(); plt.title('Depth Comparison (SVC, full extrinsic)')
    plt.savefig(depth_scatter); plt.close()
    print(f"深度散点图已保存至: {depth_scatter}")

    rel_errors = np.abs(est_depths - gt_depths) / gt_depths
    print(f"\n==== 深度误差结果 (SVC 光心, 全外参) ====")
    print(f"有效点数量: {len(valid_points_info)}")
    print(f"相对误差均值: {np.mean(rel_errors):.4f}")
    print(f"相对误差中位数: {np.median(rel_errors):.4f}")
    print(f"相对误差标准差: {np.std(rel_errors):.4f}")

    plt.figure()
    plt.hist(rel_errors, bins=30, edgecolor='black')
    plt.xlabel('Relative Error'); plt.ylabel('Count')
    plt.title('Depth Error Distribution (SVC)')
    plt.savefig('error_histogram.png'); plt.close()
    print("误差直方图已保存至 error_histogram.png")

    with open(comparison_csv, 'w') as f:
        f.write("u_s,v_s,est_depth_svc(m),gt_depth_svc(m),abs_error(m),rel_error\n")
        for (u, v, d_est, d_gt) in valid_points_info:
            abs_err = abs(d_est - d_gt)
            rel_err = abs_err / d_gt
            f.write(f"{u},{v},{d_est:.4f},{d_gt:.4f},{abs_err:.4f},{rel_err:.4f}\n")
    print(f"深度对比数据已保存至: {comparison_csv}")

    print("\n前20个有效点对比:")
    print("   u    v   est_depth  gt_depth  abs_err  rel_err")
    for i, (u, v, d_est, d_gt) in enumerate(valid_points_info[:20]):
        abs_err = abs(d_est - d_gt)
        rel_err = abs_err / d_gt
        print(f"{i:2d}: {u:4d} {v:4d}   {d_est:7.4f}   {d_gt:7.4f}   "
              f"{abs_err:7.4f}   {rel_err:6.4f}")
else:
    print("无有效投影点！")
