#!/bin/bash
# 批量评测 VIW-Fusion output 下所有 scene (SVC 版, 全外参深度评测)
# 用法: bash batch_eval_svc.sh

VIW_DIR="/home/nio/vscode/python_project_backup/VIW-Fusion"
OUTPUT_DIR="${VIW_DIR}/output"
DATA_ROOT_BASE="/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/data"
EVAL_SCRIPT="${VIW_DIR}/eval/eval_depth_svc.py"
RESULT_DIR="${OUTPUT_DIR}/eval_tmp"

mkdir -p "${RESULT_DIR}"

SUMMARY="${OUTPUT_DIR}/summary_svc.csv"
echo "scene,timestamp,valid_points,slope,rel_err_mean,rel_err_median,rel_err_std" > "${SUMMARY}"

count=0
skipped=0

for scene_dir in "${OUTPUT_DIR}"/scene_*; do
    [ -d "$scene_dir" ] || continue
    scene_name=$(basename "$scene_dir")
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

    # 读取第三行的时间戳
    timestamp=$(sed -n '3p' "$pose_file" | awk '{print $1}')
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
    tmp_py="${RESULT_DIR}/_eval_svc_${scene_name}.py"
    cat > "$tmp_py" << PYEOF
import sys, os
os.chdir("${scene_dir}")
scene_name = "${scene_name}"

# 覆盖配置变量
data_root = "${data_root}/"
pose_file = "${pose_file}"
points_file = "${points_file}"
frame_timestamp = ${timestamp}
output_image = "./projected_points_frame.png"
overlay_image = "./color_depth_overlay.png"
depth_scatter = "./depth_scatter.png"
comparison_csv = "./depth_comparison.csv"

import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# D455 深度内参 (处理后 SVC 针孔图同内参)
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
# D455 IMU -> cam0 外参 (cam0.yaml)
R_bc = np.array([
    [ 0.999994,  0.000078,  0.003535],
    [-0.000097,  0.999987,  0.005146],
    [-0.003535, -0.005147,  0.999981]
])
t_bc = np.array([-0.028933, -0.007453, -0.015558])
R_ic = R_bc.T
t_ic = -R_ic @ t_bc
cam_subdir = "svc_processed"
depth_subdir = "depth0"
DEPTH_MAX = 3.0

# === 以下逻辑从 eval_depth_svc.py 复制 ===
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

def find_cam_image(cam_dir, frame_timestamp):
    if os.path.isfile(os.path.join(cam_dir, "data.csv")):
        stamps, names = [], []
        with open(os.path.join(cam_dir, "data.csv")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split(',')
                try:
                    stamps.append(int(parts[0]) * 1e-9)
                    names.append(parts[1])
                except ValueError: continue
    else:
        stamps, names = [], []
        for p in glob.glob(os.path.join(cam_dir, "*.png")):
            base = os.path.splitext(os.path.basename(p))[0]
            ms = int(base) // 10**6
            stamps.append(ms * 1e-3)
            names.append(os.path.basename(p))
    i = int(np.argmin(np.abs(np.array(stamps) - frame_timestamp)))
    return names[i], stamps[i]

# 1. 目标帧位姿
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

# 2. 处理后的 SVC 图
cam_dir = os.path.join(data_root, cam_subdir)
cam_filename, cam_ts = find_cam_image(cam_dir, frame_timestamp)
color_img = cv2.imread(os.path.join(cam_dir, "data", cam_filename))
if color_img is None:
    print(f"ERROR: 无法读取 SVC 图: {os.path.join(cam_dir, cam_filename)}")
    sys.exit(1)
h, w = color_img.shape[:2]

# 3. 深度图
depth_csv = os.path.join(data_root, depth_subdir, "data.csv")
stamps, names = [], []
with open(depth_csv) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split(',')
        try:
            stamps.append(int(parts[0]) * 1e-9)
            names.append(parts[1])
        except ValueError: continue
i = int(np.argmin(np.abs(np.array(stamps) - frame_timestamp)))
depth_img = cv2.imread(os.path.join(data_root, depth_subdir, "data", names[i]), cv2.IMREAD_UNCHANGED)
if depth_img is None:
    print("ERROR: 无法读取深度图")
    sys.exit(1)
depth_m = depth_img.astype(np.float32) / 1000.0

# 4. 点云
points, obs_list = load_points_with_visibility(points_file)
print(f"总点数: {len(points)}")
if len(points) == 0:
    print("ERROR: 点云文件为空！")
    sys.exit(1)

# 5. 投影与深度比较 (全外参, 处理视差)
fx, fy, cx, cy = K_D[0, 0], K_D[1, 1], K_D[0, 2], K_D[1, 2]
all_pixels = []
valid_points_info = []
for i, P_w in enumerate(points):
    if obs_list[i] and not is_point_visible(frame_timestamp, obs_list[i]):
        continue
    P_i = R_wi.T @ (P_w - p_wi)
    P_p = R_c.T @ (P_i - t_c)
    if P_p[2] <= 0 or P_p[2] >= DEPTH_MAX:
        continue
    P_dc = R_sd @ P_p + t_sd
    u_d = fx * P_dc[0] / P_dc[2] + cx
    v_d = fy * P_dc[1] / P_dc[2] + cy
    u_di, v_di = int(round(u_d)), int(round(v_d))
    if not (0 <= u_di < w and 0 <= v_di < h):
        continue
    z_gt_d = depth_m[v_di, u_di]
    if z_gt_d <= 0 or z_gt_d >= DEPTH_MAX:
        continue
    P_gt_d = np.array([(u_d - cx) / fx * z_gt_d,
                       (v_d - cy) / fy * z_gt_d,
                       z_gt_d])
    P_gt_s = R_sd.T @ (P_gt_d - t_sd)
    if P_gt_s[2] <= 0:
        continue
    gt_depth = P_gt_s[2]
    est_depth = P_p[2]
    u_s = fx * P_p[0] / est_depth + cx
    v_s = fy * P_p[1] / est_depth + cy
    u_si, v_si = int(round(u_s)), int(round(v_s))
    if not (0 <= u_si < w and 0 <= v_si < h):
        continue
    all_pixels.append((u_si, v_si))
    valid_points_info.append((u_si, v_si, est_depth, gt_depth))

print(f"总投影点（SVC 图内）: {len(all_pixels)}")
print(f"有效深度点: {len(valid_points_info)}")

# 6. 投影点图
img_draw = color_img.copy()
for (u, v, d_est, d_gt) in valid_points_info:
    cv2.circle(img_draw, (u, v), 3, (0, 0, 255), -1)
cv2.putText(img_draw, f"Valid: {len(valid_points_info)}/{len(points)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
cv2.imwrite(output_image, img_draw)

# 7. SVC-深度叠加图
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_m, alpha=255.0/DEPTH_MAX), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(color_img, 0.6, depth_colormap, 0.4, 0)
cv2.imwrite(overlay_image, overlay)

# 8. 像素分布图
plt.figure(figsize=(8,6))
if all_pixels:
    plt.scatter([p[0] for p in all_pixels], [p[1] for p in all_pixels], s=1, alpha=0.3, label='all projected')
if valid_points_info:
    plt.scatter([p[0] for p in valid_points_info], [p[1] for p in valid_points_info], s=2, c='red', label='valid depth')
plt.xlim(0, w); plt.ylim(h, 0)
plt.xlabel('u'); plt.ylabel('v')
plt.legend(); plt.title('Projected Points Distribution (SVC)')
plt.savefig('pixel_distribution_detailed.png'); plt.close()

# 9. 深度误差统计
if valid_points_info:
    est_depths = np.array([d[2] for d in valid_points_info])
    gt_depths  = np.array([d[3] for d in valid_points_info])
    slope = np.linalg.lstsq(gt_depths.reshape(-1,1), est_depths, rcond=None)[0][0]
    print(f"深度回归斜率 (est/gt) = {slope:.4f}")

    plt.figure()
    plt.scatter(gt_depths, est_depths, alpha=0.6)
    plt.plot([0, max(gt_depths)], [0, slope*max(gt_depths)], 'r--', label=f'slope={slope:.2f}')
    plt.plot([0, max(gt_depths)], [0, max(gt_depths)], 'g--', label='ideal (slope=1)')
    plt.xlabel('GT Depth from SVC (m)'); plt.ylabel('VINS Est Depth from SVC (m)')
    plt.legend(); plt.title(f'{scene_name} Depth Comparison (SVC)')
    plt.savefig(depth_scatter); plt.close()

    rel_errors = np.abs(est_depths - gt_depths) / gt_depths
    print(f"==== 深度误差结果 (SVC 光心, 全外参) ====")
    print(f"有效点数量: {len(valid_points_info)}")
    print(f"相对误差均值: {np.mean(rel_errors):.4f}")
    print(f"相对误差中位数: {np.median(rel_errors):.4f}")
    print(f"相对误差标准差: {np.std(rel_errors):.4f}")

    plt.figure()
    plt.hist(rel_errors, bins=30, edgecolor='black')
    plt.xlabel('Relative Error'); plt.ylabel('Count')
    plt.title('Depth Error Distribution (SVC)')
    plt.savefig('error_histogram.png'); plt.close()

    with open(comparison_csv, 'w') as f:
        f.write("u_s,v_s,est_depth_svc(m),gt_depth_svc(m),abs_error(m),rel_error\n")
        for (u, v, d_est, d_gt) in valid_points_info:
            abs_err = abs(d_est - d_gt)
            rel_err = abs_err / d_gt
            f.write(f"{u},{v},{d_est:.4f},{d_gt:.4f},{abs_err:.4f},{rel_err:.4f}\n")

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
