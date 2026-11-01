#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从车辆 IMU 合并文件 (combined.txt) 中按每个 scene 的 RGB 时间戳范围截取
车辆 IMU 数据，写入每个 scene 的 imu1/data.csv。

combined.txt 格式 (tab 分隔带表头):
  elapsedRealtimeNanos  wallClockNs  ptpTimestampUs
  accX_m_s2 accY_m_s2 accZ_m_s2 gyroX_rad_s gyroY_rad_s gyroZ_rad_s
时间戳取 wallClockNs 列 (与轮速、D455 同一墙上时钟域)。

输出格式 (与 D455 的 imu0/data.csv 一致):
  timestamp_ns,gx,gy,gz,ax,ay,az
"""

import os
import glob

# ==================== 配置 ====================
DATA_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/data"
IMU_FILE = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/车采数据/imu/1/combined.txt"
MARGIN_NS = int(0.2 * 1e9)  # 截取时间范围比RGB前后各扩展0.2秒 (与轮速脚本一致)
# ==============================================


def load_imu_data(filepath):
    """手动解析 combined.txt，避免 float64 精度丢失 19 位时间戳。
    返回 (ts_ns, rows)，rows 每行为 [gx,gy,gz,ax,ay,az]"""
    print(f"加载车辆IMU文件: {filepath}")
    timestamps = []
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                # 第2列为 wallClockNs；跳过表头等非数字行
                ts_i = int(parts[1])
            except ValueError:
                continue
            acc = [float(x) for x in parts[3:6]]
            gyro = [float(x) for x in parts[6:9]]
            timestamps.append(ts_i)
            # 输出顺序与 D455 imu0/data.csv 一致: gx,gy,gz,ax,ay,az
            rows.append(gyro + acc)
    ts = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    ts_arr = [timestamps[i] for i in ts]
    rows_arr = [rows[i] for i in ts]
    print(f"  共 {len(ts_arr)} 条IMU记录")
    print(f"  时间范围: {ts_arr[0]} ~ {ts_arr[-1]}")
    return ts_arr, rows_arr


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


def main():
    # 加载全部车辆IMU数据
    ts_all, rows_all = load_imu_data(IMU_FILE)

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

        # 截取IMU数据
        indices = [i for i, t in enumerate(ts_all) if t_start <= t <= t_end]

        if len(indices) == 0:
            print(f"  [{scene_name}] 跳过: 时间范围内无IMU数据 "
                  f"(RGB: {t_min}~{t_max})")
            skipped += 1
            continue

        # 创建 imu1 目录
        imu_dir = os.path.join(scene_dir, "imu1")
        os.makedirs(imu_dir, exist_ok=True)

        # 写入 data.csv
        csv_path = os.path.join(imu_dir, "data.csv")
        with open(csv_path, 'w') as f:
            f.write("#timestamp [ns],gx,gy,gz,ax,ay,az\n")
            for i in indices:
                gx, gy, gz, ax, ay, az = rows_all[i]
                f.write(f"{ts_all[i]},{gx:.9f},{gy:.9f},{gz:.9f},"
                        f"{ax:.9f},{ay:.9f},{az:.9f}\n")

        print(f"  [{scene_name}] RGB: {t_min}~{t_max} | "
              f"IMU1: {ts_all[indices[0]]}~{ts_all[indices[-1]]} | "
              f"{len(indices)} 条记录")
        processed += 1

    print(f"\n完成: {processed} 个 scene 已处理, {skipped} 个跳过")


if __name__ == "__main__":
    main()
