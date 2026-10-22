#!/bin/bash
# 批量评测 VIW-Fusion output 下所有 1cam_init_scale_scene_N
# 用法: bash batch_eval_4m.sh

VIW_DIR="/home/nio/vscode/python_project_backup/VIW-Fusion"
OUTPUT_DIR="${VIW_DIR}/output"
DATA_ROOT_BASE="/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/data"
EVAL_SCRIPT="${VIW_DIR}/eval/eval_depth_4m_wheel.py"
RESULT_DIR="${OUTPUT_DIR}/eval_tmp"

mkdir -p "${RESULT_DIR}"

SUMMARY="${OUTPUT_DIR}/summary.csv"
echo "scene,timestamp,valid_points,slope,rel_err_mean,rel_err_median,rel_err_std" > "${SUMMARY}"

count=0
skipped=0

for scene_dir in "${OUTPUT_DIR}"/scene_*; do
    [ -d "$scene_dir" ] || continue
    scene_name=$(basename "$scene_dir")
    # 提取 scene 编号: 1cam_init_scale_scene_42 → scene_42
    scene_num="${scene_name#scene_}"
    data_scene="scene_${scene_num}"

    pose_file="${scene_dir}/log/pose_output.txt"
    points_file="${scene_dir}/log/points_output.txt"

    # 跳过不存在或为空的pose文件
    if [ ! -f "$pose_file" ] || [ ! -s "$pose_file" ]; then
        echo "[SKIP] ${scene_name}: pose_output.txt 为空或不存在"
        skipped=$((skipped + 1))
        continue
    fi

    if [ ! -f "$points_file" ]; then
        echo "[SKIP] ${scene_name}: points_output.txt 不存在"
        skipped=$((skipped + 1))
        continue
    fi

    # 读取最后一行的时间戳
    timestamp=$(tail -n 1 "$pose_file" | awk '{print $1}')
    if [ -z "$timestamp" ]; then
        echo "[SKIP] ${scene_name}: 无法读取时间戳"
        skipped=$((skipped + 1))
        continue
    fi

    data_root="${DATA_ROOT_BASE}/${data_scene}"
    if [ ! -d "$data_root" ]; then
        echo "[SKIP] ${scene_name}: 数据目录不存在 ${data_root}"
        skipped=$((skipped + 1))
        continue
    fi

    echo "=============================================="
    echo "[${count}] 评测 ${scene_name} → ${data_scene}  timestamp=${timestamp}"
    echo "=============================================="

    # 生成临时Python脚本，覆盖配置后执行eval
    tmp_py="${RESULT_DIR}/_eval_${scene_name}.py"
    cat > "$tmp_py" << PYEOF
import sys, os
os.chdir("${scene_dir}")
scene_name = "${scene_name}"

# 覆盖配置变量
data_root = "${data_root}"
pose_file = "${pose_file}"
points_file = "${points_file}"
frame_timestamp = ${timestamp}
output_image = "./projected_points_frame.png"
overlay_image = "./color_depth_overlay.png"
depth_scatter = "./depth_scatter.png"
comparison_csv = "./depth_comparison.csv"

# 读取原始脚本，跳过配置区域，注入我们的变量
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = np.array([
    [381.2039, 0,        320.0508],
    [0,        380.8693, 241.8335],
    [0,        0,        1       ]
])
R_body_cam = np.array([
    [ 0.999994,  0.000078,  0.003535],
    [-0.000097,  0.999987,  0.005146],
    [-0.003535, -0.005147,  0.999981]
])
t_body_cam = np.array([-0.028933, -0.007453, -0.015558])
R_ic = R_body_cam.T
t_ic = -R_ic @ t_body_cam
cam_subdir = "cam0"
depth_subdir = "depth0_undistorted"

# === 以下逻辑从 eval_depth_4m_wheel.py 复制 ===
def parse_pose_line(line):
    parts = line.strip().split()
    t = float(parts[0])
    p = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    return t, p, q

def quat_to_rot(q):
    qx, qy, qz, qw = q
    R = np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw  ],
        [2*qx*qy+2*qz*qw,   1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw  ],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx**2-2*qy**2]
    ])
    return R

def load_points_with_visibility(pf):
    points, obs = [], []
    with open(pf, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = list(map(float, line.split()))
            if len(parts) < 3: continue
            points.append(parts[:3])
            obs.append(parts[3:] if len(parts) > 3 else [])
    return np.array(points), obs

def is_point_visible(target_stamp, obs_list, tol=1e-4):
    for s in obs_list:
        if abs(s - target_stamp) < tol: return True
    return False

# 1. 读取目标帧IMU位姿
target_pose = None
with open(pose_file, 'r') as f:
    for line in f:
        t, p_wi, q_wi = parse_pose_line(line)
        if abs(t - frame_timestamp) < 1e-4:
            target_pose = (p_wi, q_wi)
            break
if target_pose is None:
    print(f"ERROR: 未找到时间戳 {frame_timestamp}")
    sys.exit(1)
p_wi, q_wi = target_pose
R_wi = quat_to_rot(q_wi)

# 2. 读取彩色图
cam_csv = os.path.join(data_root, cam_subdir, "data.csv")
df_cam = pd.read_csv(cam_csv)
df_cam['timestamp_sec'] = df_cam.iloc[:, 0] * 1e-9
idx_cam = (df_cam['timestamp_sec'] - frame_timestamp).abs().idxmin()
cam_filename = df_cam.iloc[idx_cam, 1]
cam_img_path = os.path.join(data_root, cam_subdir, "data", cam_filename)
color_img = cv2.imread(cam_img_path)
if color_img is None:
    print(f"ERROR: 无法读取彩色图: {cam_img_path}")
    sys.exit(1)
h, w = color_img.shape[:2]

# 3. 读取深度图
depth_csv = os.path.join(data_root, depth_subdir, "data.csv")
df_depth = pd.read_csv(depth_csv)
df_depth['timestamp_sec'] = df_depth.iloc[:, 0] * 1e-9
idx_depth = (df_depth['timestamp_sec'] - frame_timestamp).abs().idxmin()
depth_filename = df_depth.iloc[idx_depth, 1]
depth_img_path = os.path.join(data_root, depth_subdir, "data", depth_filename)
depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
if depth_img is None:
    print(f"ERROR: 无法读取深度图: {depth_img_path}")
    sys.exit(1)
depth_m = depth_img.astype(np.float32) / 1000.0

# 4. 加载点云
points, obs_list = load_points_with_visibility(points_file)
print(f"总点数: {len(points)}")
if len(points) == 0:
    print("ERROR: 点云文件为空！")
    sys.exit(1)

# 5. 投影和深度比较
all_pixels = []
valid_points_info = []
for i, P_w in enumerate(points):
    if obs_list[i] and not is_point_visible(frame_timestamp, obs_list[i]):
        continue
    P_i = R_wi.T @ (P_w - p_wi)
    P_c = R_ic @ P_i + t_ic
    depth_est = P_c[2]
    if depth_est <= 0 or depth_est >= 2.5:
        continue
    u = K[0,0] * P_c[0] / depth_est + K[0,2]
    v = K[1,1] * P_c[1] / depth_est + K[1,2]
    u_int, v_int = int(round(u)), int(round(v))
    if 0 <= u_int < w and 0 <= v_int < h:
        all_pixels.append((u_int, v_int))
        depth_gt = depth_m[v_int, u_int]
        if depth_gt > 0 and depth_gt < 2.5:
            valid_points_info.append((u_int, v_int, depth_est, depth_gt))

print(f"\n总投影点（图像内）: {len(all_pixels)}")
print(f"有效深度点: {len(valid_points_info)}")

# 6. 绘制投影点
img_draw = color_img.copy()
for (u, v, d_est, d_gt) in valid_points_info:
    cv2.circle(img_draw, (u, v), 3, (0, 0, 255), -1)
cv2.putText(img_draw, f"Valid: {len(valid_points_info)}/{len(points)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
cv2.imwrite(output_image, img_draw)
print(f"投影图像已保存至: {output_image}")

# 7. 彩色+深度叠加图
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_m, alpha=255.0/10.0), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(color_img, 0.6, depth_colormap, 0.4, 0)
cv2.imwrite(overlay_image, overlay)
print(f"彩色-深度叠加图已保存至: {overlay_image}")

# 8. 像素分布图
plt.figure(figsize=(8,6))
if all_pixels:
    u_vals = [p[0] for p in all_pixels]
    v_vals = [p[1] for p in all_pixels]
    plt.scatter(u_vals, v_vals, s=1, alpha=0.3, label='all projected')
if valid_points_info:
    u_val = [p[0] for p in valid_points_info]
    v_val = [p[1] for p in valid_points_info]
    plt.scatter(u_val, v_val, s=2, c='red', label='valid depth')
plt.xlim(0, w)
plt.ylim(h, 0)
plt.xlabel('u'); plt.ylabel('v')
plt.legend(); plt.title('Projected Points Distribution')
plt.savefig('pixel_distribution_detailed.png'); plt.close()
print("详细像素分布图已保存至 pixel_distribution_detailed.png")

# 9. 深度误差统计
if valid_points_info:
    est_depths = np.array([d[2] for d in valid_points_info])
    gt_depths  = np.array([d[3] for d in valid_points_info])
    slope = np.linalg.lstsq(gt_depths.reshape(-1,1), est_depths, rcond=None)[0][0]
    print(f"\n深度回归斜率 (est/gt) = {slope:.4f}")

    # 散点图
    plt.figure()
    plt.scatter(gt_depths, est_depths, alpha=0.6)
    plt.plot([0, max(gt_depths)], [0, slope*max(gt_depths)], 'r--', label=f'slope={slope:.2f}')
    plt.plot([0, max(gt_depths)], [0, max(gt_depths)], 'g--', label='ideal (slope=1)')
    plt.xlabel('GT Depth (m)'); plt.ylabel('VINS Est Depth (m)')
    plt.legend(); plt.title(f'{scene_name} Depth Comparison')
    plt.savefig(depth_scatter); plt.close()
    print(f"深度散点图已保存至: {depth_scatter}")

    # 相对误差
    rel_errors = np.abs(est_depths - gt_depths) / gt_depths
    print(f"\n==== 深度误差结果 ====")
    print(f"有效点数量: {len(valid_points_info)}")
    print(f"相对误差均值: {np.mean(rel_errors):.4f}")
    print(f"相对误差中位数: {np.median(rel_errors):.4f}")
    print(f"相对误差标准差: {np.std(rel_errors):.4f}")

    # 误差直方图
    plt.figure()
    plt.hist(rel_errors, bins=30, edgecolor='black')
    plt.xlabel('Relative Error'); plt.ylabel('Count')
    plt.title('Depth Error Distribution')
    plt.savefig('error_histogram.png'); plt.close()
    print("误差直方图已保存至 error_histogram.png")

    # 保存详细CSV
    with open(comparison_csv, 'w') as f:
        f.write("u,v,est_depth(m),gt_depth(m),abs_error(m),rel_error\n")
        for (u, v, d_est, d_gt) in valid_points_info:
            abs_err = abs(d_est - d_gt)
            rel_err = abs_err / d_gt
            f.write(f"{u},{v},{d_est:.4f},{d_gt:.4f},{abs_err:.4f},{rel_err:.4f}\n")
    print(f"深度对比数据已保存至: {comparison_csv}")

    # 打印前20个点
    print("\n前20个有效点对比:")
    print("   u    v   est_depth  gt_depth  abs_err  rel_err")
    for i, (u, v, d_est, d_gt) in enumerate(valid_points_info[:20]):
        abs_err = abs(d_est - d_gt)
        rel_err = abs_err / d_gt
        print(f"{i:2d}: {u:4d} {v:4d}   {d_est:7.4f}   {d_gt:7.4f}   {abs_err:7.4f}   {rel_err:6.4f}")

    # 写结果到stdout最后一行，供shell脚本解析
    print(f"RESULT:{len(valid_points_info)},{slope:.6f},{np.mean(rel_errors):.6f},{np.median(rel_errors):.6f},{np.std(rel_errors):.6f}")
else:
    print("无有效投影点！")
    print("RESULT:0,0,0,0,0")
PYEOF

    # 运行评测
    output=$(python3 "$tmp_py" 2>&1) || true
    echo "$output" | tail -5

    # 提取结果行写入summary
    result_line=$(echo "$output" | grep "^RESULT:" | tail -1)
    if [ -n "$result_line" ]; then
        result_data="${result_line#RESULT:}"
        echo "${scene_name},${timestamp},${result_data}" >> "${SUMMARY}"
    else
        echo "${scene_name},${timestamp},FAIL" >> "${SUMMARY}"
    fi

    count=$((count + 1))
done

echo ""
echo "=============================================="
echo "评测完成: ${count} 个场景, 跳过 ${skipped} 个"
echo "汇总文件: ${SUMMARY}"
echo "=============================================="
cat "${SUMMARY}"
