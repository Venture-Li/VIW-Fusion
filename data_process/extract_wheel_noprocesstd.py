#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 combined.txt 中按每个 scene 的 RGB 时间戳范围截取轮速信息，
转换为 VINS 格式 (timestamp_ns,vx,vy,vz,wx,wy,wz) 写入 wheel0/data.csv。

combined.txt 格式: timestamp_ns  FL(km/h)  FR(km/h)  RL(km/h)  RR(km/h)
输出格式:           timestamp_ns,vx,vy,vz,wx,wy,wz

转换公式 (后轮差速模型, B_r = 1.720m):
  vx = (v_RL + v_RR) / 2 * (1000/3600)   # km/h -> m/s
  wz = (v_RL - v_RR) / B_r * (1000/3600) # km/h -> m/s -> rad/s
  vy = vz = wx = wy = 0
"""

import os
import glob
import numpy as np

# ==================== 配置 ====================
DATA_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/data"
SPEED_FILE = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/车采数据/轮速/1/combined.txt"
B_R = 1.720  # 后轮距 (米), 实测轮距 1720mm
MARGIN_NS = int(0.2 * 1e9)  # 轮速时间范围略大于RGB: 前后各扩展0.2秒
# ==============================================

KMH_TO_MS = 1000.0 / 3600.0


def load_speed_data(filepath):
    """手动解析 combined.txt，避免 float64 精度丢失 19 位时间戳"""
    print(f"加载轮速文件: {filepath}")
    timestamps = []
    speeds = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                # 跳过表头等非数字行 (combined.txt 首行为 wallClockNs 列名)
                ts_i = int(parts[0])
            except ValueError:
                continue
            timestamps.append(ts_i)
            speeds.append([float(x) for x in parts[1:5]])
    ts = np.array(timestamps, dtype=np.int64)
    speeds = np.array(speeds, dtype=np.float64)
    print(f"  共 {len(ts)} 条轮速记录")
    print(f"  时间范围: {ts[0]} ~ {ts[-1]}")
    return ts, speeds


def get_rgb_timestamp_range(scene_dir):
    """读取 cam0/data.csv，返回 (min_ts, max_ts)"""
    csv_path = os.path.join(scene_dir, "cam0", "data.csv")
    if not os.path.exists(csv_path):
        return None, None

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    timestamps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) >= 1:
            try:
                timestamps.append(int(parts[0]))
            except ValueError:
                continue

    if not timestamps:
        return None, None

    return min(timestamps), max(timestamps)


def convert_wheel(speeds_kmh):
    """
    四轮速度 (FL, FR, RL, RR) km/h -> (vx, vy, vz, wx, wy, wz) m/s, rad/s
    后轮差速模型:
      vx = (v_RL + v_RR) / 2
      wz = (v_RL - v_RR) / B_r
    """
    v_RL = speeds_kmh[2] * KMH_TO_MS
    v_RR = speeds_kmh[3] * KMH_TO_MS

    vx = (v_RL + v_RR) / 2.0
    vy = 0.0
    vz = 0.0
    wx = 0.0
    wy = 0.0
    wz = (v_RL - v_RR) / B_R

    return vx, vy, vz, wx, wy, wz


def main():
    # 加载全部轮速数据
    ts_all, speeds_all = load_speed_data(SPEED_FILE)

    # 获取所有 scene 目录
    scene_dirs = sorted(
        glob.glob(os.path.join(DATA_DIR, "scene_*")),
        key=lambda x: int(x.split("_")[-1])
    )
    print(f"\n找到 {len(scene_dirs)} 个 scene 目录")

    processed = 0
    skipped = 0

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        t_min, t_max = get_rgb_timestamp_range(scene_dir)

        if t_min is None or t_max is None:
            print(f"  [{scene_name}] 跳过: 无 cam0/data.csv")
            skipped += 1
            continue

        # 扩展时间范围
        t_start = t_min - MARGIN_NS
        t_end = t_max + MARGIN_NS

        # 截取轮速数据
        mask = (ts_all >= t_start) & (ts_all <= t_end)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"  [{scene_name}] 跳过: 时间范围内无轮速数据 "
                  f"(RGB: {t_min}~{t_max})")
            skipped += 1
            continue

        ts_scene = ts_all[indices]
        speeds_scene = speeds_all[indices]

        # 创建 wheel0 目录
        wheel_dir = os.path.join(scene_dir, "wheel0")
        os.makedirs(wheel_dir, exist_ok=True)

        # 写入 data.csv
        csv_path = os.path.join(wheel_dir, "data.csv")
        with open(csv_path, 'w') as f:
            f.write("#timestamp [ns],vx,vy,vz,wx,wy,wz\n")
            for i in range(len(ts_scene)):
                vx, vy, vz, wx, wy, wz = convert_wheel(speeds_scene[i])
                f.write(f"{ts_scene[i]},{vx:.6f},{vy:.6f},{vz:.6f},"
                        f"{wx:.6f},{wy:.6f},{wz:.6f}\n")

        print(f"  [{scene_name}] RGB: {t_min}~{t_max} | "
              f"Wheel: {ts_scene[0]}~{ts_scene[-1]} | "
              f"{len(ts_scene)} 条记录")
        processed += 1

    print(f"\n完成: {processed} 个 scene 已处理, {skipped} 个跳过")


if __name__ == "__main__":
    main()
