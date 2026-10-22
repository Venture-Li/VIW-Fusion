#!/bin/bash
BASE_DIR="/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/data/动态/"

echo "=== 开始复制 depth0/data.csv 到 depth0_undistorted/ ==="
for scene in "$BASE_DIR"/scene_*; do
    src="$scene/depth0/data.csv"
    dst_dir="$scene/depth0_undistorted"
    dst="$dst_dir/data.csv"

    if [ ! -f "$src" ]; then
        echo "警告: $src 不存在，跳过"
        continue
    fi

    if [ ! -d "$dst_dir" ]; then
        echo "警告: $dst_dir 目录不存在，跳过"
        continue
    fi

    if [ -f "$dst" ]; then
        echo "跳过（已存在）: $dst"
    else
        cp "$src" "$dst_dir/"
        echo "已复制: $src -> $dst_dir/"
    fi
done

echo "完成。"