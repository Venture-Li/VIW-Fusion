#!/bin/bash
# copy_bag_to_container.sh

SOURCE_DIR="/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/bag"
CONTAINER="viw-fusion-container"
DEST_DIR="/root/scene"

# 1. 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ 错误：源目录不存在 -> $SOURCE_DIR"
    exit 1
fi

# 2. 检查容器是否存在（无论是否运行）
if ! docker inspect "$CONTAINER" &>/dev/null; then
    echo "❌ 错误：容器 $CONTAINER 不存在"
    exit 1
fi

# 3. 在容器内创建目标目录（如果不存在则创建）
echo "📁 在容器内创建目录 $DEST_DIR ..."
docker exec "$CONTAINER" mkdir -p "$DEST_DIR"

# 4. 使用 tar 管道复制所有内容（不保留 bag 目录名，直接将内容放入 scene）
echo "📦 正在复制 $SOURCE_DIR 下的所有内容到 $CONTAINER:$DEST_DIR ..."
tar -cf - -C "$SOURCE_DIR" . | docker exec -i "$CONTAINER" tar -xf - -C "$DEST_DIR"

# 5. 验证（可选）
echo "✅ 复制完成！容器内文件列表："
docker exec "$CONTAINER" ls -la "$DEST_DIR"