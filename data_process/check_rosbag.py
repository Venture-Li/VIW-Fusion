#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比官方 ridgeback_light.bag 和我们合成的 scene_*.bag 的各项差异。
用法:
  /home/nio/miniforge3/envs/da/bin/python3 check_rosbag.py
"""

import sys
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ==================== 配置 ====================
OFFICIAL_BAG = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/ridgeback/ridgeback_light.bag"
OUR_BAG = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/time/scene_time_processtd.bag"
# ==============================================

SEP = "=" * 70
MINOR_SEP = "-" * 50


def load_bag_info(bag_path, label):
    """加载 bag 的基本信息：topic、消息数、时间范围、msgtype"""
    info = {}
    print(f"\n{SEP}")
    print(f"  {label}: {bag_path}")
    print(f"{SEP}")

    with AnyReader([Path(bag_path)]) as reader:
        for conn in reader.connections:
            topic = conn.topic
            msgtype = conn.msgtype
            count = 0
            first_ts = last_ts = None
            first_raw = None
            for _, ts, raw in reader.messages(connections=[conn]):
                if first_ts is None:
                    first_ts = ts
                    first_raw = raw
                last_ts = ts
                count += 1
            if count > 0:
                info[topic] = {
                    'count': count,
                    'first_ts': first_ts,
                    'last_ts': last_ts,
                    'duration': (last_ts - first_ts) / 1e9,
                    'msgtype': msgtype,
                    'first_raw': first_raw,
                }
                print(f"  [{topic}]")
                print(f"    msgtype:    {msgtype}")
                print(f"    count:      {count}")
                print(f"    first_ts:   {first_ts} ({first_ts/1e9:.6f}s)")
                print(f"    last_ts:    {last_ts} ({last_ts/1e9:.6f}s)")
                print(f"    duration:   {(last_ts-first_ts)/1e9:.3f}s")
    return info


def compare_topics(official_info, our_info):
    """对比 topic 列表"""
    print(f"\n{SEP}")
    print(f"  TOPIC 对比")
    print(f"{SEP}")

    official_topics = set(official_info.keys())
    our_topics = set(our_info.keys())

    # VIW-Fusion 实际使用的 topic
    used_topics = {
        '/camera/infra1/image_rect_raw',
        '/camera/infra1/camera_info',
        '/camera/imu',
        '/ridgeback_velocity_controller/odom',
    }

    print(f"\n  VIW-Fusion 订阅的 topic:")
    for t in sorted(used_topics):
        in_official = "✓" if t in official_topics else "✗"
        in_ours = "✓" if t in our_topics else "✗"
        print(f"    [{in_official}] official  [{in_ours}] ours  {t}")

    only_official = official_topics - our_topics
    only_ours = our_topics - official_topics
    if only_official:
        print(f"\n  仅官方有: {sorted(only_official)}")
    if only_ours:
        print(f"\n  仅我们有: {sorted(only_ours)}")


def compare_msg_counts(official_info, our_info):
    """对比消息频率"""
    print(f"\n{SEP}")
    print(f"  消息频率对比")
    print(f"{SEP}")

    common_topics = set(official_info.keys()) & set(our_info.keys())
    for topic in sorted(common_topics):
        o = official_info[topic]
        u = our_info[topic]
        o_hz = o['count'] / o['duration'] if o['duration'] > 0 else 0
        u_hz = u['count'] / u['duration'] if u['duration'] > 0 else 0
        print(f"  [{topic}]")
        print(f"    official: {o['count']} msgs / {o['duration']:.1f}s = {o_hz:.1f} Hz")
        print(f"    ours:     {u['count']} msgs / {u['duration']:.1f}s = {u_hz:.1f} Hz")


def compare_imu_content(official_bag, our_bag):
    """深入对比 IMU 消息内容"""
    print(f"\n{SEP}")
    print(f"  IMU 消息内容对比 (/camera/imu)")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/camera/imu':
                    samples = []
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        hdr_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                        ax = msg.linear_acceleration.x
                        ay = msg.linear_acceleration.y
                        az = msg.linear_acceleration.z
                        gx = msg.angular_velocity.x
                        gy = msg.angular_velocity.y
                        gz = msg.angular_velocity.z
                        samples.append({
                            'record_ts': ts,
                            'header_ts': hdr_ts,
                            'diff_ns': ts - hdr_ts,
                            'acc': (ax, ay, az),
                            'gyr': (gx, gy, gz),
                            'frame_id': msg.header.frame_id,
                            'ori_cov': msg.orientation_covariance[0] if len(msg.orientation_covariance) > 0 else None,
                            'ang_cov': msg.angular_velocity_covariance[0] if len(msg.angular_velocity_covariance) > 0 else None,
                            'lin_cov': msg.linear_acceleration_covariance[0] if len(msg.linear_acceleration_covariance) > 0 else None,
                        })
                        if i >= 10:
                            break

                    # 前5条
                    print(f"    前5条消息:")
                    for i, s in enumerate(samples[:5]):
                        print(f"      [{i}] record_ts={s['record_ts']} ({s['record_ts']/1e9:.6f}s)")
                        print(f"          header.stamp={s['header_ts']} ({s['header_ts']/1e9:.6f}s)")
                        print(f"          diff(record-header)={s['diff_ns']}ns = {s['diff_ns']/1e9:.6f}s")
                        print(f"          acc=({s['acc'][0]:.4f}, {s['acc'][1]:.4f}, {s['acc'][2]:.4f})")
                        print(f"          gyr=({s['gyr'][0]:.4f}, {s['gyr'][1]:.4f}, {s['gyr'][2]:.4f})")
                        print(f"          frame_id='{s['frame_id']}'")
                        print(f"          ori_cov[0]={s['ori_cov']}, ang_cov[0]={s['ang_cov']}, lin_cov[0]={s['lin_cov']}")

                    # 统计
                    acc_x = [s['acc'][0] for s in samples]
                    acc_y = [s['acc'][1] for s in samples]
                    acc_z = [s['acc'][2] for s in samples]
                    gyr_x = [s['gyr'][0] for s in samples]
                    gyr_y = [s['gyr'][1] for s in samples]
                    gyr_z = [s['gyr'][2] for s in samples]
                    diffs = [s['diff_ns'] for s in samples]
                    print(f"\n    统计 (前{len(samples)}条):")
                    print(f"      acc_x:  [{min(acc_x):.4f}, {max(acc_x):.4f}]")
                    print(f"      acc_y:  [{min(acc_y):.4f}, {max(acc_y):.4f}]")
                    print(f"      acc_z:  [{min(acc_z):.4f}, {max(acc_z):.4f}]")
                    print(f"      gyr_x:  [{min(gyr_x):.4f}, {max(gyr_x):.4f}]")
                    print(f"      gyr_y:  [{min(gyr_y):.4f}, {max(gyr_y):.4f}]")
                    print(f"      gyr_z:  [{min(gyr_z):.4f}, {max(gyr_z):.4f}]")
                    print(f"      record_ts - header.stamp: [{min(diffs)/1e9:.6f}s, {max(diffs)/1e9:.6f}s]")
                    break


def compare_image_content(official_bag, our_bag):
    """深入对比图像消息内容"""
    print(f"\n{SEP}")
    print(f"  图像消息内容对比 (/camera/infra1/image_rect_raw)")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/camera/infra1/image_rect_raw':
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        hdr_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                        print(f"    第{i+1}帧:")
                        print(f"      record_ts:     {ts} ({ts/1e9:.6f}s)")
                        print(f"      header.stamp:  {hdr_ts} ({hdr_ts/1e9:.6f}s)")
                        print(f"      diff:          {(ts-hdr_ts)/1e9:.6f}s")
                        print(f"      frame_id:      '{msg.header.frame_id}'")
                        print(f"      size:          {msg.width}x{msg.height}")
                        print(f"      encoding:      '{msg.encoding}'")
                        print(f"      step:          {msg.step}")
                        print(f"      is_bigendian:  {msg.is_bigendian}")
                        print(f"      data_len:      {len(msg.data)} bytes")
                        if i >= 2:
                            break
                    break


def compare_camera_info(official_bag, our_bag):
    """对比 CameraInfo"""
    print(f"\n{SEP}")
    print(f"  CameraInfo 对比 (/camera/infra1/camera_info)")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/camera/infra1/camera_info':
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        hdr_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                        print(f"    第{i+1}条:")
                        print(f"      record_ts:        {ts} ({ts/1e9:.6f}s)")
                        print(f"      header.stamp:     {hdr_ts} ({hdr_ts/1e9:.6f}s)")
                        print(f"      frame_id:         '{msg.header.frame_id}'")
                        print(f"      size:             {msg.width}x{msg.height}")
                        print(f"      distortion_model: '{msg.distortion_model}'")
                        print(f"      K:                {list(msg.K)}")
                        print(f"      D:                {list(msg.D)}")
                        print(f"      R:                {list(msg.R)}")
                        print(f"      P:                {list(msg.P)}")
                        if i >= 0:
                            break
                    break


def compare_wheel_content(official_bag, our_bag):
    """对比轮速/Odometry 消息"""
    print(f"\n{SEP}")
    print(f"  Wheel/Odometry 对比 (/ridgeback_velocity_controller/odom)")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/ridgeback_velocity_controller/odom':
                    samples = []
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        hdr_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                        samples.append({
                            'record_ts': ts,
                            'header_ts': hdr_ts,
                            'diff_ns': ts - hdr_ts,
                            'frame_id': msg.header.frame_id,
                            'child_frame_id': msg.child_frame_id,
                            'vx': msg.twist.twist.linear.x,
                            'vy': msg.twist.twist.linear.y,
                            'vz': msg.twist.twist.linear.z,
                            'wx': msg.twist.twist.angular.x,
                            'wy': msg.twist.twist.angular.y,
                            'wz': msg.twist.twist.angular.z,
                            'pose_cov_0': msg.pose.covariance[0],
                            'twist_cov_0': msg.twist.covariance[0],
                        })
                        if i >= 10:
                            break

                    print(f"    前5条消息:")
                    for i, s in enumerate(samples[:5]):
                        print(f"      [{i}] record_ts={s['record_ts']} ({s['record_ts']/1e9:.6f}s)")
                        print(f"          header.stamp={s['header_ts']} ({s['header_ts']/1e9:.6f}s)")
                        print(f"          diff(record-header)={s['diff_ns']/1e9:.6f}s")
                        print(f"          frame_id='{s['frame_id']}', child='{s['child_frame_id']}'")
                        print(f"          v=({s['vx']:.4f}, {s['vy']:.4f}, {s['vz']:.4f})")
                        print(f"          w=({s['wx']:.4f}, {s['wy']:.4f}, {s['wz']:.4f})")
                        print(f"          pose_cov[0]={s['pose_cov_0']}, twist_cov[0]={s['twist_cov_0']}")

                    vx_vals = [s['vx'] for s in samples]
                    wz_vals = [s['wz'] for s in samples]
                    print(f"\n    统计 (前{len(samples)}条):")
                    print(f"      vx: [{min(vx_vals):.4f}, {max(vx_vals):.4f}]")
                    print(f"      wz: [{min(wz_vals):.4f}, {max(wz_vals):.4f}]")
                    break


def compare_timestamp_alignment(official_bag, our_bag):
    """对比各传感器时间戳对齐情况"""
    print(f"\n{SEP}")
    print(f"  传感器时间戳对齐分析")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            first_ts = {}
            for conn in reader.connections:
                topic = conn.topic
                if topic not in ['/camera/infra1/image_rect_raw', '/camera/imu',
                                 '/ridgeback_velocity_controller/odom']:
                    continue
                for _, ts, raw in reader.messages(connections=[conn]):
                    first_ts[topic] = ts
                    break

        if first_ts:
            base = min(first_ts.values())
            print(f"    基准时间戳 (最早): {base} ({base/1e9:.6f}s)")
            for topic in sorted(first_ts.keys()):
                offset = (first_ts[topic] - base) / 1e9
                print(f"    {topic}: offset={offset:+.6f}s")


def compare_imu_header_vs_record(official_bag, our_bag):
    """关键对比：IMU 的 record timestamp vs header.stamp"""
    print(f"\n{SEP}")
    print(f"  ★ 关键: IMU record_ts vs header.stamp 差异")
    print(f"  (VIW-Fusion 使用 header.stamp.toSec())")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/camera/imu':
                    diffs = []
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        hdr_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                        diffs.append((ts - hdr_ts) / 1e9)
                        if i >= 1000:
                            break
                    diffs = np.array(diffs)
                    print(f"    record_ts - header.stamp (秒):")
                    print(f"      min:    {diffs.min():.6f}")
                    print(f"      max:    {diffs.max():.6f}")
                    print(f"      mean:   {diffs.mean():.6f}")
                    print(f"      std:    {diffs.std():.6f}")
                    print(f"      median: {np.median(diffs):.6f}")
                    break


def compare_wheel_header_vs_record(official_bag, our_bag):
    """关键对比：Wheel 的 record timestamp vs header.stamp"""
    print(f"\n{SEP}")
    print(f"  ★ 关键: Wheel record_ts vs header.stamp 差异")
    print(f"  (VIW-Fusion 使用 header.stamp.toSec())")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/ridgeback_velocity_controller/odom':
                    diffs = []
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        hdr_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                        diffs.append((ts - hdr_ts) / 1e9)
                        if i >= 1000:
                            break
                    diffs = np.array(diffs)
                    print(f"    record_ts - header.stamp (秒):")
                    print(f"      min:    {diffs.min():.6f}")
                    print(f"      max:    {diffs.max():.6f}")
                    print(f"      mean:   {diffs.mean():.6f}")
                    print(f"      std:    {diffs.std():.6f}")
                    break


def compare_image_header_vs_record(official_bag, our_bag):
    """关键对比：图像 record timestamp vs header.stamp"""
    print(f"\n{SEP}")
    print(f"  ★ 关键: Image record_ts vs header.stamp 差异")
    print(f"  (VIW-Fusion 使用 header.stamp.toSec())")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/camera/infra1/image_rect_raw':
                    diffs = []
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        hdr_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                        diffs.append((ts - hdr_ts) / 1e9)
                        if i >= 100:
                            break
                    diffs = np.array(diffs)
                    print(f"    record_ts - header.stamp (秒):")
                    print(f"      min:    {diffs.min():.6f}")
                    print(f"      max:    {diffs.max():.6f}")
                    print(f"      mean:   {diffs.mean():.6f}")
                    print(f"      std:    {diffs.std():.6f}")
                    break


def compare_imu_acc_range(official_bag, our_bag):
    """对比 IMU 加速度数值范围（检查单位是否一致）"""
    print(f"\n{SEP}")
    print(f"  ★ IMU 加速度数值范围 (检查单位: m/s²)")
    print(f"{SEP}")

    for label, bag_path in [("OFFICIAL", official_bag), ("OURS", our_bag)]:
        print(f"\n  --- {label} ---")
        with AnyReader([Path(bag_path)]) as reader:
            for conn in reader.connections:
                if conn.topic == '/camera/imu':
                    acc_norms = []
                    gyr_norms = []
                    for i, (_, ts, raw) in enumerate(reader.messages(connections=[conn])):
                        msg = reader.deserialize(raw, conn.msgtype)
                        ax = msg.linear_acceleration.x
                        ay = msg.linear_acceleration.y
                        az = msg.linear_acceleration.z
                        gx = msg.angular_velocity.x
                        gy = msg.angular_velocity.y
                        gz = msg.angular_velocity.z
                        acc_norms.append(np.sqrt(ax*ax + ay*ay + az*az))
                        gyr_norms.append(np.sqrt(gx*gx + gy*gy + gz*gz))
                        if i >= 5000:
                            break
                    acc_norms = np.array(acc_norms)
                    gyr_norms = np.array(gyr_norms)
                    print(f"    acc norm:  min={acc_norms.min():.4f}, max={acc_norms.max():.4f}, mean={acc_norms.mean():.4f}")
                    print(f"    gyr norm:  min={gyr_norms.min():.4f}, max={gyr_norms.max():.4f}, mean={gyr_norms.mean():.4f}")
                    print(f"    (静止时 acc_norm 应接近 9.8 m/s²)")
                    break


def main():
    print("=" * 70)
    print("  VIW-Fusion ROS Bag 对比分析")
    print("=" * 70)

    # 1. 基本信息
    official_info = load_bag_info(OFFICIAL_BAG, "OFFICIAL")
    our_info = load_bag_info(OUR_BAG, "OURS")

    # 2. Topic 对比
    compare_topics(official_info, our_info)

    # 3. 消息频率对比
    compare_msg_counts(official_info, our_info)

    # 4. 时间戳对齐
    compare_timestamp_alignment(OFFICIAL_BAG, OUR_BAG)

    # 5. record_ts vs header.stamp (关键!)
    compare_imu_header_vs_record(OFFICIAL_BAG, OUR_BAG)
    compare_wheel_header_vs_record(OFFICIAL_BAG, OUR_BAG)
    compare_image_header_vs_record(OFFICIAL_BAG, OUR_BAG)

    # 6. IMU 内容对比
    compare_imu_content(OFFICIAL_BAG, OUR_BAG)

    # 7. IMU 数值范围
    compare_imu_acc_range(OFFICIAL_BAG, OUR_BAG)

    # 8. 图像内容对比
    compare_image_content(OFFICIAL_BAG, OUR_BAG)

    # 9. CameraInfo 对比
    compare_camera_info(OFFICIAL_BAG, OUR_BAG)

    # 10. Wheel 内容对比
    compare_wheel_content(OFFICIAL_BAG, OUR_BAG)

    print(f"\n{SEP}")
    print(f"  对比完成")
    print(f"{SEP}")


if __name__ == "__main__":
    main()