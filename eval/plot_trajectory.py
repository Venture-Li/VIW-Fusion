#!/usr/bin/env python3
"""可视化 VINS-simple vs VIW-Fusion 轨迹对比
- 仅使用两个算法共同覆盖的时间段内的帧
- 按时间戳最近邻匹配后做 Umeyama 对齐
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径 ==========
ref_path = "/home/nio/vscode/python_project_backup/VIW-Fusion/output/2cam_ori/vio.csv"
est_path = "/home/nio/vscode/python_project_backup/VIW-Fusion/output/1cam_init/vio.csv"
output_path = "/home/nio/vscode/python_project_backup/VIW-Fusion/output/1cam_init/eval/trajectory_comparison.png"

# ========== 加载（统一将纳秒时间戳转为秒） ==========
def load_viw_fusion(path):
    stamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            t = float(parts[0]) / 1e9   # 纳秒 → 秒
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            stamps.append(t)
            poses.append([x, y, z])
    return np.array(stamps), np.array(poses)

def load_vins_simple(path):
    stamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            t = float(parts[0]) / 1e9   # 纳秒 → 秒
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            stamps.append(t)
            poses.append([x, y, z])
    return np.array(stamps), np.array(poses)

ref_stamps, ref_xyz = load_viw_fusion(ref_path)
est_stamps, est_xyz = load_vins_simple(est_path)

print(f"VIW-Fusion: {len(ref_stamps)} 帧, {ref_stamps[0]:.3f}s ~ {ref_stamps[-1]:.3f}s")
print(f"VINS-simple: {len(est_stamps)} 帧, {est_stamps[0]:.3f}s ~ {est_stamps[-1]:.3f}s")

# ========== 裁剪到共同时间范围 ==========
t_start = max(ref_stamps[0], est_stamps[0])
t_end = min(ref_stamps[-1], est_stamps[-1])
print(f"\n共同时间范围: {t_start:.3f}s ~ {t_end:.3f}s ({t_end - t_start:.1f}s)")

ref_mask = (ref_stamps >= t_start) & (ref_stamps <= t_end)
est_mask = (est_stamps >= t_start) & (est_stamps <= t_end)

ref_stamps_crop = ref_stamps[ref_mask]
ref_xyz_crop = ref_xyz[ref_mask]
est_stamps_crop = est_stamps[est_mask]
est_xyz_crop = est_xyz[est_mask]

print(f"VIW-Fusion 共同帧: {len(ref_stamps_crop)}")
print(f"VINS-simple 共同帧: {len(est_stamps_crop)}")

# ========== 坐标系统诊断 ==========
print(f"\n=== 坐标系统诊断 ===")
print(f"VIW-Fusion 范围: X[{ref_xyz_crop[:,0].min():.2f}, {ref_xyz_crop[:,0].max():.2f}] "
      f"Y[{ref_xyz_crop[:,1].min():.2f}, {ref_xyz_crop[:,1].max():.2f}] "
      f"Z[{ref_xyz_crop[:,2].min():.2f}, {ref_xyz_crop[:,2].max():.2f}]")
print(f"VINS-simple 范围: X[{est_xyz_crop[:,0].min():.2f}, {est_xyz_crop[:,0].max():.2f}] "
      f"Y[{est_xyz_crop[:,1].min():.2f}, {est_xyz_crop[:,1].max():.2f}] "
      f"Z[{est_xyz_crop[:,2].min():.2f}, {est_xyz_crop[:,2].max():.2f}]")

# ========== 按时间戳最近邻匹配（阈值 0.05 秒） ==========
def align_by_timestamp(ref_t, ref_p, est_t, est_p, max_dt=0.05):
    """对 est 的每一帧，在 ref 中找最近时间戳匹配"""
    aligned_ref, aligned_est, matched_t = [], [], []
    for i, t in enumerate(est_t):
        idx = np.argmin(np.abs(ref_t - t))
        if abs(ref_t[idx] - t) < max_dt:
            aligned_ref.append(ref_p[idx])
            aligned_est.append(est_p[i])
            matched_t.append(t)
    return np.array(aligned_ref), np.array(aligned_est), np.array(matched_t)

ref_aligned, est_aligned, matched_t = align_by_timestamp(ref_stamps_crop, ref_xyz_crop,
                                                          est_stamps_crop, est_xyz_crop)
print(f"\n时间戳匹配帧数: {len(ref_aligned)}")

if len(ref_aligned) == 0:
    raise RuntimeError("没有匹配到任何帧，请检查时间戳单位或放宽阈值！")

# ========== Umeyama 对齐 (est -> ref) ==========
def umeyama_alignment(P, Q):
    """计算 (s, R, t) 使得 s*R*P + t ≈ Q"""
    n = P.shape[0]
    mu_p = np.mean(P, axis=0)
    mu_q = np.mean(Q, axis=0)
    sigma_p = np.mean(np.sum((P - mu_p)**2, axis=1))
    cov = (Q - mu_q).T @ (P - mu_p) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_p if sigma_p > 1e-10 else 1.0
    t = mu_q - s * R @ mu_p
    return s, R, t

s, R, t = umeyama_alignment(est_aligned, ref_aligned)
est_transformed = (s * (R @ est_aligned.T)).T + t

print(f"\nUmeyama 对齐结果:")
print(f"  scale:  {s:.4f}")
print(f"  rotation:\n{R}")
print(f"  translation: {t}")

# 检查旋转矩阵是否接近单位阵（坐标系是否一致）
rot_diff = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
print(f"  旋转角度: {np.degrees(rot_diff):.2f}° (0° = 坐标系一致)")

# ========== 误差计算 ==========
errors = np.sqrt(np.sum((ref_aligned - est_transformed)**2, axis=1))
rmse = np.sqrt(np.mean(errors**2))
print(f"\nATE RMSE: {rmse:.4f} m")
print(f"  均值: {np.mean(errors):.4f} m")
print(f"  中位数: {np.median(errors):.4f} m")
print(f"  最大值: {np.max(errors):.4f} m")
print(f"  标准差: {np.std(errors):.4f} m")

# ========== 绘图 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 原始轨迹（仅共同时间范围，未对齐）
ax = axes[0, 0]
ax.plot(ref_xyz_crop[:, 0], ref_xyz_crop[:, 1], 'b-', linewidth=1.0, label='VIW-Fusion (参考)', alpha=0.8)
ax.plot(est_xyz_crop[:, 0], est_xyz_crop[:, 1], 'r-', linewidth=1.0, label='VINS-simple (原始)', alpha=0.8)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('原始轨迹 (共同时间范围, 未对齐)')
ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

# 2. Umeyama 对齐后
ax = axes[0, 1]
ax.plot(ref_aligned[:, 0], ref_aligned[:, 1], 'b-', linewidth=1.0, label='VIW-Fusion (参考)', alpha=0.8)
ax.plot(est_transformed[:, 0], est_transformed[:, 1], 'r--', linewidth=1.0,
        label=f'VINS-simple (对齐, s={s:.3f}, θ={np.degrees(rot_diff):.1f}°)', alpha=0.8)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('Umeyama 对齐后')
ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

# 3. 逐帧误差 (用实际时间戳)
ax = axes[1, 0]
ax.plot(matched_t - matched_t[0], errors, 'k-', linewidth=0.5, alpha=0.7)
ax.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'mean={np.mean(errors):.2f}m')
ax.axhline(y=np.median(errors), color='g', linestyle='--', label=f'median={np.median(errors):.2f}m')
ax.set_xlabel('时间 (s)'); ax.set_ylabel('误差 (m)')
ax.set_title('逐帧 ATE 误差')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. 误差分布直方图
ax = axes[1, 1]
ax.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(x=np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'mean={np.mean(errors):.2f}m')
ax.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2, label=f'median={np.median(errors):.2f}m')
ax.set_xlabel('误差 (m)'); ax.set_ylabel('频次')
ax.set_title('误差分布')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle(f'VINS-simple vs VIW-Fusion 轨迹对比\n'
             f'ATE RMSE = {rmse:.2f}m, Umeyama Scale = {s:.3f}, Rotation = {np.degrees(rot_diff):.1f}°\n'
             f'共同帧: {len(ref_aligned)} (时间范围 {t_end - t_start:.1f}s)',
             fontsize=12)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")