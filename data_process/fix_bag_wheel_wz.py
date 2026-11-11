#!/usr/bin/env python3
"""scene6 轮速方向对照实验：只取反 Odometry.angular.z，无需 ROS。

原 bag 不变，输出存在时拒绝覆盖。其他话题（包括 joint_states）原样保留；
这是供当前订阅 Odometry 的 SLAM 做对照的 bag，不是完整轮速数据重标定。
"""

from pathlib import Path

from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

BAG_DIR = Path(
    "/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/data/ros_bags"
)
INPUT_BAG = BAG_DIR / "scene_6_overlap.bag"
OUTPUT_BAG = BAG_DIR / "scene_6_overlap_wz_fixed.bag"
WHEEL_TOPIC = "/ridgeback_velocity_controller/odom"
WHEEL_TYPE = "nav_msgs/msg/Odometry"


def negate_wheel_wz(raw, typestore):
    message = typestore.deserialize_ros1(raw, WHEEL_TYPE)
    message.twist.twist.angular.z = -message.twist.twist.angular.z
    return typestore.serialize_ros1(message, WHEEL_TYPE)


def main():
    if not INPUT_BAG.is_file():
        raise FileNotFoundError(INPUT_BAG)
    if OUTPUT_BAG.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{OUTPUT_BAG}")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    changed = 0
    with Reader(INPUT_BAG) as reader:
        wheel_connections = [c for c in reader.connections if c.topic == WHEEL_TOPIC]
        if not wheel_connections or not sum(c.msgcount for c in wheel_connections):
            raise ValueError(f"缺少轮速消息：{WHEEL_TOPIC}")
        if any(c.msgtype != WHEEL_TYPE for c in wheel_connections):
            raise ValueError("轮速话题不是 nav_msgs/Odometry")

        print(f"输入：{INPUT_BAG}")
        print(f"输出：{OUTPUT_BAG}", flush=True)
        with Writer(OUTPUT_BAG) as writer:
            connections = {}
            for c in reader.connections:
                connections[c.id] = writer.add_connection(
                    c.topic, c.msgtype,
                    msgdef=c.msgdef.data, md5sum=c.digest,
                    callerid=c.ext.callerid, latching=c.ext.latching,
                )
            for connection, timestamp, raw in reader.messages():
                if connection.topic == WHEEL_TOPIC:
                    raw = negate_wheel_wz(raw, typestore)
                    changed += 1
                writer.write(connections[connection.id], timestamp, raw)

    print(f"完成：{changed} 条轮速消息的 angular.z 已取反。")
    print("消息数量、时间戳及其他字段保持不变。")


if __name__ == "__main__":
    main()
