#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 data、bag、output 三个目录中的 scene 从 1 开始连续重编号，保持对应关系。

用法:
  python3 renumber_scenes.py          # 预览映射
  python3 renumber_scenes.py --run    # 执行重命名
"""
import os
import sys

DATA_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/data"
BAG_DIR  = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/bag"
OUT_DIR  = "/home/nio/vscode/python_project_backup/VIW-Fusion/output"

TEMP_PREFIX_DATA = "scene_renumber_tmp_"
TEMP_PREFIX_BAG  = "scene_renumber_tmp_"
TEMP_PREFIX_OUT  = "1cam_init_scale_scene_renumber_tmp_"


def get_scene_numbers(data_dir):
    """获取 data 目录中所有 scene 编号（已排序）"""
    nums = []
    for d in os.listdir(data_dir):
        if d.startswith("scene_") and os.path.isdir(os.path.join(data_dir, d)):
            try:
                nums.append(int(d.split("_")[-1]))
            except ValueError:
                pass
    return sorted(nums)


def build_mapping(old_nums):
    """构建 old_num -> new_num 映射"""
    return {old: new for new, old in enumerate(old_nums, 1)}


def preview(old_nums, mapping):
    print(f"共 {len(old_nums)} 个 scene，映射关系：")
    print(f"{'旧序号':>6} -> {'新序号':>6}")
    print("-" * 18)
    for old in old_nums:
        print(f"{old:>6} -> {mapping[old]:>6}")


def do_rename(old_nums, mapping):
    """三步重命名：原始 -> 临时 -> 最终，避免冲突"""
    ops = []  # (dir_type, old_path, tmp_path, new_path)

    for old in old_nums:
        new = mapping[old]

        # data
        old_path = os.path.join(DATA_DIR, f"scene_{old}")
        tmp_path = os.path.join(DATA_DIR, f"{TEMP_PREFIX_DATA}{new}")
        new_path = os.path.join(DATA_DIR, f"scene_{new}")
        if os.path.isdir(old_path):
            ops.append(("data", old_path, tmp_path, new_path))

        # bag
        old_path = os.path.join(BAG_DIR, f"scene_{old}.bag")
        tmp_path = os.path.join(BAG_DIR, f"{TEMP_PREFIX_BAG}{new}.bag")
        new_path = os.path.join(BAG_DIR, f"scene_{new}.bag")
        if os.path.isfile(old_path):
            ops.append(("bag", old_path, tmp_path, new_path))

        # output
        old_path = os.path.join(OUT_DIR, f"1cam_init_scale_scene_{old}")
        tmp_path = os.path.join(OUT_DIR, f"{TEMP_PREFIX_OUT}{new}")
        new_path = os.path.join(OUT_DIR, f"1cam_init_scale_scene_{new}")
        if os.path.isdir(old_path):
            ops.append(("output", old_path, tmp_path, new_path))

    # 第一步：原始 -> 临时
    print("\n[1/2] 重命名为临时名称...")
    for dir_type, old_path, tmp_path, _ in ops:
        os.rename(old_path, tmp_path)
        print(f"  [{dir_type}] {os.path.basename(old_path)} -> {os.path.basename(tmp_path)}")

    # 第二步：临时 -> 最终
    print("\n[2/2] 重命名为最终名称...")
    for dir_type, _, tmp_path, new_path in ops:
        os.rename(tmp_path, new_path)
        print(f"  [{dir_type}] {os.path.basename(tmp_path)} -> {os.path.basename(new_path)}")

    print(f"\n完成！共处理 {len(ops)} 项。")


def main():
    old_nums = get_scene_numbers(DATA_DIR)
    if not old_nums:
        print("错误: data 目录中无 scene_* 文件夹")
        sys.exit(1)

    mapping = build_mapping(old_nums)

    if "--run" in sys.argv:
        do_rename(old_nums, mapping)
    else:
        preview(old_nums, mapping)
        print(f"\n预览模式。确认无误后执行: python3 {sys.argv[0]} --run")


if __name__ == "__main__":
    main()
