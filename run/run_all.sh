#!/bin/bash
# VIW-Fusion 批量运行脚本（本地运行，通过 docker exec 操作容器）
# 用法: bash /home/nio/vscode/python_project_backup/VIW-Fusion/run/run_all.sh
#
# 可视化说明:
#   脚本使用 rosbag record 录制 /vins_estimator/image_track 话题。
#   录制文件保存到 /media/nio/.../data/scene_N/track.bag。
#   跑完后在任意有 X11 的机器上回放查看特征跟踪画面。
#   如果不需要录制，设置 VISUALIZE=0

set -e

VISUALIZE="${VISUALIZE:-1}"   # 设为 0 禁用可视化录制
CONTAINER="d05978977759"
CONFIG="/root/catkin_ws/src/VIW-Fusion/config/realsense_d455i/realsense_stereo_imu_config_d455.yaml"
OUTPUT_DIR="/root/catkin_ws/src/VIW-Fusion/output"
BAG_DIR="/root/scene"
SETUP="source /opt/ros/melodic/setup.bash && source /root/catkin_ws/devel/setup.bash"

# 宿主机上 track bag 的目标目录
HOST_DATA_BASE="/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/data"
HOST_OUTPUT="/home/nio/vscode/python_project_backup/VIW-Fusion/output"

# 期望的输出文件（不含时间戳，用通配符匹配）
EXPECTED_FILES=(
    "extrinsic_iterate_wheel.csv"
    "extrinsic_parameter.csv"
    "groundtruth.csv"
    "points_output.txt"
    "pose_output.txt"
    "process_time.csv"
    "vins_init_log_*.txt"
    "vins_points_log_*.txt"
    "vio.csv"
)

# 确保 roscore 在容器中运行
if ! docker exec "$CONTAINER" bash -c "$SETUP && rostopic list" &>/dev/null; then
    echo "Starting roscore in container..."
    docker exec -d "$CONTAINER" bash -c "$SETUP && roscore" &
    sleep 3
fi

for i in $(seq 1 20); do
    bag_file="$BAG_DIR/scene_$i.bag"
    if ! docker exec "$CONTAINER" test -f "$bag_file" 2>/dev/null; then
        echo "SKIP: $bag_file not found"
        continue
    fi

    log_dir="$OUTPUT_DIR/scene_$i/log"
    docker exec "$CONTAINER" mkdir -p "$log_dir"

    echo "=== [$(date '+%H:%M:%S')] Processing scene_$i ==="

    # ★ 启动前清理：确保 OUTPUT_DIR 根目录下无残留文件
    docker exec "$CONTAINER" bash -c "find $OUTPUT_DIR -maxdepth 1 -type f \( -name '*.txt' -o -name '*.csv' \) -delete" 2>/dev/null || true

    # 启动 VINS 节点（后台）
    docker exec -d "$CONTAINER" bash -c "$SETUP && rosrun vins viwo_node $CONFIG"

    sleep 2

    # 启动可视化录制（后台，录制 image_track 话题到 bag）
    if [ "$VISUALIZE" = "1" ]; then
        track_bag="$OUTPUT_DIR/track.bag"
        docker exec -d "$CONTAINER" bash -c "$SETUP && rosbag record -O $track_bag /vins_estimator/image_track __name:=track_recorder" 2>/dev/null
        echo "Track recorder started → $track_bag"
    fi

    # 播放 bag（阻塞直到播放完毕）
    docker exec "$CONTAINER" bash -c "$SETUP && rosbag play --clock $bag_file"

    # 轮询等待所有输出文件生成完毕
    echo "Waiting for output files..."
    MAX_WAIT=60
    waited=0
    while true; do
        all_ready=true
        for pattern in "${EXPECTED_FILES[@]}"; do
            count=$(docker exec "$CONTAINER" bash -c "find $OUTPUT_DIR -maxdepth 1 -type f -name '$pattern' | wc -l")
            if [ "$count" -eq 0 ]; then
                all_ready=false
                break
            fi
        done
        if $all_ready; then
            echo "All output files ready."
            break
        fi
        if [ $waited -ge $MAX_WAIT ]; then
            echo "WARNING: Timeout waiting for output files after ${MAX_WAIT}s, proceeding anyway."
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done

    # 终止 VINS 节点
    docker exec "$CONTAINER" bash -c "$SETUP && rosnode kill /viwo_node" 2>/dev/null || true

    # ★ 等待节点进程彻底退出（而非仅 sleep 固定时间）
    echo "Waiting for viwo_node to exit..."
    for _ in $(seq 1 30); do
        if ! docker exec "$CONTAINER" pgrep -f "viwo_node" &>/dev/null; then
            break
        fi
        sleep 1
    done
    # 如果还没退出，强杀
    if docker exec "$CONTAINER" pgrep -f "viwo_node" &>/dev/null; then
        echo "WARNING: viwo_node still alive, force killing..."
        docker exec "$CONTAINER" bash -c "kill -9 \$(pgrep -f viwo_node)" 2>/dev/null || true
        sleep 1
    fi
    echo "viwo_node exited."

    # 终止 track_recorder 并移动 track.bag 到 data 目录
    if [ "$VISUALIZE" = "1" ]; then
        docker exec "$CONTAINER" bash -c "$SETUP && rosnode kill /track_recorder" 2>/dev/null || true
        sleep 1
        host_track_bag="$HOST_OUTPUT/track.bag"
        target_dir="$HOST_DATA_BASE/scene_$i"
        if [ -f "$host_track_bag" ]; then
            mkdir -p "$target_dir"
            mv "$host_track_bag" "$target_dir/track.bag"
            echo "Track bag saved → $target_dir/track.bag"
        fi
    fi

    # 移动 output 根目录下的 txt/csv 到对应 scene 的 log 子目录
    docker exec "$CONTAINER" bash -c "find $OUTPUT_DIR -maxdepth 1 -type f \( -name '*.txt' -o -name '*.csv' \) -exec mv {} $log_dir/ \;"

    echo "=== [$(date '+%H:%M:%S')] scene_$i done ==="
done

echo "=== All scenes processed ==="
