#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import pandas as pd

def main():
    base = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21"
    svc_source = os.path.join(base, "svc")
    data_root = os.path.join(base, "data", "静态_区分时延_D455imu")

    if not os.path.isdir(svc_source):
        print(f"错误：源图片目录不存在: {svc_source}")
        return

    for i in range(1, 65):
        scene_name = f"scene_{i}"
        scene_dir = os.path.join(data_root, scene_name)
        imu_csv = os.path.join(scene_dir, "imu0", "data.csv")

        if not os.path.isfile(imu_csv):
            print(f"警告：{imu_csv} 不存在，跳过")
            continue

        # 自动检测分隔符（逗号、制表符或空格）
        try:
            df = pd.read_csv(imu_csv, sep=None, engine='python')
        except Exception as e:
            print(f"错误：读取 {imu_csv} 失败: {e}")
            continue

        if df.empty:
            print(f"警告：{imu_csv} 为空，跳过")
            continue

        # 提取第一列（时间戳），并确保转换为整数
        try:
            start_ns = int(pd.to_numeric(df.iloc[0, 0]))
            end_ns = int(pd.to_numeric(df.iloc[-1, 0]))
        except Exception as e:
            print(f"错误：无法解析时间戳列: {e}")
            continue

        print(f"处理 {scene_name}: 时间范围 [{start_ns}, {end_ns}]")

        # 创建目标 svc 子文件夹
        target_svc = os.path.join(scene_dir, "svc")
        os.makedirs(target_svc, exist_ok=True)

        # 匹配图片文件名
        pattern = re.compile(r'^CAMERA0_FRAME_(\d+)\.jpg$')
        copied_count = 0
        for filename in os.listdir(svc_source):
            match = pattern.match(filename)
            if not match:
                continue
            ts_ms = int(match.group(1))          # 毫秒
            ts_ns = ts_ms * 1_000_000            # 转换为纳秒

            if start_ns <= ts_ns <= end_ns:
                src_path = os.path.join(svc_source, filename)
                dst_path = os.path.join(target_svc, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1

        print(f"  复制了 {copied_count} 张图片到 {target_svc}")

if __name__ == "__main__":
    main()