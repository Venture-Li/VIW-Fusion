#!/usr/bin/env python3
"""只裁剪 scene_6 的图像时间范围；用 viw-bag 环境运行，无需 ROS。

输入、输出路径以及时间偏移见下方常量。原 bag 不变，输出已存在时拒绝覆盖。
图像及对应 camera_info 按共同覆盖范围筛选；IMU、轮速和其余话题原样保留，
因此保留了初始化、积分和插值需要的边界样本。不会修改消息内容或记录时间。
这是首尾覆盖对照实验，不检查/修复传感器内部长间隔和标定正确性。
"""

from pathlib import Path
import struct

from rosbags.rosbag1 import Reader, Writer

BAG_DIR = Path(
    "/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/data/ros_bags"
)
INPUT_BAG = BAG_DIR / "scene_6.bag"
OUTPUT_BAG = BAG_DIR / "scene_6_overlap.bag"

IMAGE_TOPIC = "/camera/infra1/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/infra1/camera_info"
IMU_TOPIC = "/camera/imu"
WHEEL_TOPIC = "/ridgeback_velocity_controller/odom"

# 必须与本次运行的 YAML 一致；使用整数纳秒计算，避免大时间戳浮点精度损失。
# IMU 对应时间 = image_stamp + td
# 轮速对应时间 = image_stamp + td - td_wheel
TD_NS = -48_000_000
TD_WHEEL_NS = 0


def header_stamp_ns(raw):
    """以上四种 ROS1 消息都以 Header 开头：uint32 seq, uint32 sec, uint32 nsec。"""
    if len(raw) < 12:
        raise ValueError("消息过短，无法读取 ROS1 Header")
    _, sec, nsec = struct.unpack_from("<III", raw)
    if nsec >= 1_000_000_000:
        raise ValueError("非法 Header 纳秒字段")
    return sec * 1_000_000_000 + nsec


def format_time(ns):
    return f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d}"


def main():
    if not INPUT_BAG.is_file():
        raise FileNotFoundError(INPUT_BAG)
    if OUTPUT_BAG.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{OUTPUT_BAG}")

    # 第一遍只收集 Header 时间，不解码图像。
    times = {IMAGE_TOPIC: [], IMU_TOPIC: [], WHEEL_TOPIC: []}
    with Reader(INPUT_BAG) as reader:
        expected_types = {
            IMAGE_TOPIC: "sensor_msgs/msg/Image",
            CAMERA_INFO_TOPIC: "sensor_msgs/msg/CameraInfo",
            IMU_TOPIC: "sensor_msgs/msg/Imu",
            WHEEL_TOPIC: "nav_msgs/msg/Odometry",
        }
        for connection in reader.connections:
            expected = expected_types.get(connection.topic)
            if expected and connection.msgtype != expected:
                raise ValueError(f"{connection.topic} 消息类型不匹配")
        connections = [c for c in reader.connections if c.topic in times]
        for connection, _, raw in reader.messages(connections=connections):
            times[connection.topic].append(header_stamp_ns(raw))

    for topic, stamps in times.items():
        if not stamps:
            raise ValueError(f"缺少数据：{topic}")
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"存在重复或倒序时间戳，请先检查：{topic}")

    imu_start, imu_end = times[IMU_TOPIC][0], times[IMU_TOPIC][-1]
    wheel_start, wheel_end = times[WHEEL_TOPIC][0], times[WHEEL_TOPIC][-1]
    image_start = max(imu_start - TD_NS, wheel_start - TD_NS + TD_WHEEL_NS)
    image_end = min(imu_end - TD_NS, wheel_end - TD_NS + TD_WHEEL_NS)
    images = times[IMAGE_TOPIC]
    kept = [t for t in images if image_start <= t <= image_end]
    if not kept:
        raise ValueError("没有同时被 IMU 和轮速覆盖的图像，不生成输出")
    selected_stamps = set(kept)

    print(f"输入：{INPUT_BAG}")
    print(f"IMU 范围： {format_time(imu_start)} ~ {format_time(imu_end)}")
    print(f"轮速范围： {format_time(wheel_start)} ~ {format_time(wheel_end)}")
    print(f"保留图像： {format_time(kept[0])} ~ {format_time(kept[-1])}")
    print(f"图像总数：{len(images)}，保留：{len(kept)}")
    print(f"去掉开头：{sum(t < image_start for t in images)} 张")
    print(f"去掉末尾：{sum(t > image_end for t in images)} 张")
    print("写入新 bag；IMU、轮速及其他话题完整保留……", flush=True)

    # 第二遍复制原始序列化消息；所有保留消息的 Header 和 bag 记录时间均不变。
    counts = {}
    with Reader(INPUT_BAG) as reader, Writer(OUTPUT_BAG) as writer:
        connections = {}
        for connection in reader.connections:
            connections[connection.id] = writer.add_connection(
                connection.topic,
                connection.msgtype,
                msgdef=connection.msgdef.data,
                md5sum=connection.digest,
                callerid=connection.ext.callerid,
                latching=connection.ext.latching,
            )
        for connection, timestamp, raw in reader.messages():
            if connection.topic in (IMAGE_TOPIC, CAMERA_INFO_TOPIC):
                if header_stamp_ns(raw) not in selected_stamps:
                    continue
            writer.write(connections[connection.id], timestamp, raw)
            counts[connection.topic] = counts.get(connection.topic, 0) + 1

    print("写入完成，bag 已关闭。")
    for topic, count in sorted(counts.items()):
        print(f"  {topic}: {count} 条")
    print(f"输出：{OUTPUT_BAG}")


if __name__ == "__main__":
    main()
