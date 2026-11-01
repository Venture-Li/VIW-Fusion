#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import pandas as pd

def main():
    base = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18"
    svc_source = os.path.join(base, "车采数据", "CAMERA0")
    data_root = os.path.join(base, "data")

    if not os.path.isdir(svc_source):
        print(f"错误：源图片目录不存在: {svc_source}")
        return

    # 遍历实际存在的 scene 目录（删除/重编号后自动适配）
    scene_dirs = sorted(
        [d for d in os.listdir(data_root)
         if d.startswith("scene_") and os.path.isdir(os.path.join(data_root, d))],
        key=lambda x: int(x.split("_")[-1])
    )

    for scene_name in scene_dirs:
        scene_dir = os.path.join(data_root, scene_name)
        cam_csv = os.path.join(scene_dir, "cam0", "data.csv")

        if not os.path.isfile(cam_csv):
            print(f"警告：{cam_csv} 不存在，跳过")
            continue

        # 读取 cam0/data.csv 第一列时间戳，取首尾作为该 scene 时间范围
        start_ns = None
        end_ns = None
        with open(cam_csv, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    ts = int(line.split(',')[0])
                except ValueError:
                    continue
                if start_ns is None:
                    start_ns = ts
                end_ns = ts

        if start_ns is None:
            print(f"警告：{cam_csv} 无有效时间戳，跳过")
            continue

        print(f"处理 {scene_name}: 时间范围 [{start_ns}, {end_ns}]")

        # 创建目标 svc 子文件夹
        target_svc = os.path.join(scene_dir, "svc")
        os.makedirs(target_svc, exist_ok=True)

        # 匹配图片文件名 (毫秒时间戳)
        pattern = re.compile(r'^CAMERA0_FRAME_(\d+)\.jpg$')
        moved_count = 0
        for filename in os.listdir(svc_source):
            match = pattern.match(filename)
            if not match:
                continue
            ts_ms = int(match.group(1))          # 毫秒
            ts_ns = ts_ms * 1_000_000            # 转换为纳秒

            if start_ns <= ts_ns <= end_ns:
                src_path = os.path.join(svc_source, filename)
                dst_path = os.path.join(target_svc, filename)
                shutil.move(src_path, dst_path)
                moved_count += 1

        print(f"  移动了 {moved_count} 张图片到 {target_svc}")

if __name__ == "__main__":
    main()