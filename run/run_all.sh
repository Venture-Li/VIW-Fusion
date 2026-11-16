#!/bin/bash
# 在宿主机运行；通过 docker exec 使用容器内的 ROS 环境。
# 用法: bash /home/nio/code/VIW-Fusion/run/run_all.sh
# VISUALIZE=0 可关闭 image_track 录制。
set -euo pipefail

VISUALIZE="${VISUALIZE:-1}"
CONTAINER="viw-dev"
CONFIG="/root/catkin_ws/src/VIW-Fusion/config/realsense_d455i/realsense_stereo_imu_config_d455.yaml"
OUTPUT_DIR="/root/catkin_ws/src/VIW-Fusion/output"
BAG_DIR="/data/ros_bags_wz"
SETUP="source /opt/ros/melodic/setup.bash && source /root/catkin_ws/devel/setup.bash"
HOST_DATA_BASE="/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/results/track"
HOST_OUTPUT="/home/nio/code/VIW-Fusion/output"

# 等订阅注册完成再播放，sleep 仅用于轮询间隔，不判断处理是否结束。
wait_for_subscriber() {
    local topic="$1" node="$2" process_pid="$3"
    until docker exec "$CONTAINER" bash -c "$SETUP && exec python -c '
import rosgraph, sys
subscriptions = rosgraph.Master(sys.argv[2] + \"_wait\").getSystemState()[1]
sys.exit(0 if any(t == sys.argv[1] and sys.argv[2] in nodes for t, nodes in subscriptions) else 1)
' \"\$@\"" _ "$topic" "$node" 2>/dev/null; do
        if ! kill -0 "$process_pid" 2>/dev/null; then
            wait "$process_pid" || true
            echo "ERROR: $node exited before subscribing; see $log_dir."
            return 1
        fi
        sleep 0.2
    done
}

# 已有一轮在运行时，不清理其文件，也不启动同名节点。
if docker exec "$CONTAINER" pgrep -x viwo_node >/dev/null; then
    echo "ERROR: viwo_node is already running; leave its output untouched."
    exit 1
fi
if ! docker exec "$CONTAINER" bash -c "$SETUP && rostopic list" &>/dev/null; then
    docker exec -d "$CONTAINER" bash -c "$SETUP && exec roscore"
    until docker exec "$CONTAINER" bash -c "$SETUP && rostopic list" &>/dev/null; do
        sleep 0.2
    done
fi
if docker exec "$CONTAINER" bash -c "$SETUP && rosnode list" | grep -qx '/track_recorder'; then
    echo "ERROR: track_recorder is already running; leave its recording untouched."
    exit 1
fi

image_topic=$(docker exec "$CONTAINER" sed -n 's/^image0_topic: *"\([^"]*\)".*/\1/p' "$CONFIG")
test -n "$image_topic"

for i in $(seq 1 20); do
    bag_file="$BAG_DIR/scene_$i.bag"
    if ! docker exec "$CONTAINER" test -f "$bag_file"; then
        echo "SKIP: $bag_file not found"
        continue
    fi

    # 使用 bag 索引中的图像总数；前端处理完这些图像才放入结束标记。
    image_count=$(docker exec "$CONTAINER" bash -c "$SETUP && exec python -c '
import rosbag, sys
with rosbag.Bag(sys.argv[1]) as bag:
    print(bag.get_message_count(topic_filters=[sys.argv[2]]))
' \"\$@\"" _ "$bag_file" "$image_topic")
    test "$image_count" -gt 0

    log_dir="$OUTPUT_DIR/scene_$i/log"
    docker exec "$CONTAINER" mkdir -p "$log_dir"
    echo "=== [$(date '+%H:%M:%S')] Processing scene_$i ($image_count images) ==="
    docker exec "$CONTAINER" find "$OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.txt' -o -name '*.csv' \) -delete

    # 保留 docker exec 的子进程句柄，wait 等到节点及其文件收尾真正完成。
    docker exec "$CONTAINER" bash -c "$SETUP && exec rosrun vins viwo_node \"\$1\" \"\$2\" > \"\$3\" 2>&1" \
        _ "$CONFIG" "$image_count" "$log_dir/viwo_node.log" &
    vins_pid=$!
    wait_for_subscriber "$image_topic" /vins_estimator "$vins_pid"

    if [ "$VISUALIZE" = "1" ]; then
        docker exec "$CONTAINER" bash -c "$SETUP && exec rosbag record -O \"\$1\" /vins_estimator/image_track __name:=track_recorder > \"\$2\" 2>&1" \
            _ "$OUTPUT_DIR/track.bag" "$log_dir/track_recorder.log" &
        recorder_pid=$!
        wait_for_subscriber /vins_estimator/image_track /track_recorder "$recorder_pid"
    fi

    docker exec "$CONTAINER" bash -c "$SETUP && exec rosbag play --clock \"\$1\"" _ "$bag_file"

    echo "Bag playback finished; waiting for frontend/backend completion..."
    if ! wait "$vins_pid"; then
        echo "ERROR: SLAM exited without completing; output has not been archived. See $log_dir/viwo_node.log."
        exit 1
    fi
    echo "SLAM exited normally; all queued frame outputs are closed."

    if [ "$VISUALIZE" = "1" ]; then
        docker exec "$CONTAINER" bash -c "$SETUP && rosnode kill /track_recorder"
        wait "$recorder_pid"
        # rosbag record 关闭并完成索引后，才把 .bag 移至目标目录。
        docker exec "$CONTAINER" test -f "$OUTPUT_DIR/track.bag"
        target_dir="$HOST_DATA_BASE/scene_$i"
        mkdir -p "$target_dir"
        mv "$HOST_OUTPUT/track.bag" "$target_dir/track.bag"
        echo "Track bag saved → $target_dir/track.bag"
    fi

    docker exec "$CONTAINER" find "$OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.txt' -o -name '*.csv' \) -exec mv {} "$log_dir/" \;
    echo "=== [$(date '+%H:%M:%S')] scene_$i done ==="
done

echo "=== All scenes processed ==="
