#!/bin/bash
BASE_DIR="/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/data/动态/"

# ---------- 第一步：复制 data.csv ----------
echo "=== 开始复制 data.csv ==="
for scene in "$BASE_DIR"/scene_*; do
    src="$scene/cam0/data.csv"
    dst_dir="$scene/cam0_undistorted"
    dst="$dst_dir/data.csv"

    if [ -f "$src" ]; then
        if [ -f "$dst" ]; then
            echo "跳过（已存在）: $dst"
        else
            cp "$src" "$dst_dir/"
            echo "已复制: $src -> $dst_dir/"
        fi
    else
        echo "警告: $src 不存在"
    fi
done

# ---------- 第二步：重命名文件夹 ----------
echo -e "\n=== 开始重命名文件夹 ==="
for scene in "$BASE_DIR"/scene_*; do
    cam0="$scene/cam0"
    cam_ori="$scene/cam_ori"
    cam0_undist="$scene/cam0_undistorted"

    # 检查必要的目录是否存在
    if [ ! -d "$cam0" ]; then
        echo "跳过 $scene : cam0 目录不存在"
        continue
    fi
    if [ ! -d "$cam0_undist" ]; then
        echo "跳过 $scene : cam0_undistorted 目录不存在"
        continue
    fi

    # 如果 cam_ori 已存在，说明之前可能执行过，跳过重命名以免覆盖
    if [ -d "$cam_ori" ]; then
        echo "跳过 $scene : cam_ori 已存在（可能已重命名过）"
        continue
    fi

    # 重命名 cam0 -> cam_ori
    mv "$cam0" "$cam_ori"
    echo "重命名: $cam0 -> $cam_ori"

    # 重命名 cam0_undistorted -> cam0
    mv "$cam0_undist" "$cam0"
    echo "重命名: $cam0_undist -> $cam0"
done

echo -e "\n所有操作完成。"