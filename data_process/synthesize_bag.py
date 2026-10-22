#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 D455 原始 bag + 去畸变图像 + 轮速 CSV → 合成 ridgeback_light 格式 bag。

输入:
  /media/nio/.../6.24/data/20260624_*.bag         原始 D455 bag
  /media/nio/.../6.24/data/scene_N/cam0/data/*.png  去畸变 RGB
  /media/nio/.../6.24/data/scene_N/wheel0/data.csv              轮速 (vx,vy,vz,wx,wy,wz)
  /media/nio/.../6.24/data/scene_N/intrinsics.txt               去畸变后内参

输出:
  /media/nio/.../6.24/bag/scene_N.bag             ridgeback_light 格式

映射:
  Color_0/image/data  → /camera/infra1/image_rect_raw  (替换为去畸变灰度图)
  Color_0/info/camera_info → /camera/infra1/camera_info (k1=k2=p1=p2=0)
  Accel_0/imu/data + Gyro_0/imu/data → /camera/imu    (合并)
  wheel0/data.csv    → /joint_states + /ridgeback_velocity_controller/odom

用法:
  /home/nio/miniforge3/envs/da/bin/python3 synthesize_bag.py
"""

import os
import sys
import glob
import struct
import numpy as np
import cv2
from pathlib import Path

# ==================== 配置 ====================
SRC_BAG_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/6.24/data"
DATA_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/data"
OUTPUT_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/bag/"
B_R = 1.672  # 后轮距 (米)
# ==============================================


def get_camera_info_from_bag(src_bag_path):
    """从原始 bag 读取 CameraInfo，提取 K 矩阵和分辨率"""
    from rosbags.highlevel import AnyReader
    with AnyReader([Path(src_bag_path)]) as reader:
        for conn in reader.connections:
            if conn.topic == '/device_0/sensor_1/Color_0/info/camera_info':
                for _, ts, raw in reader.messages(connections=[conn]):
                    msg = reader.deserialize(raw, conn.msgtype)
                    return {
                        'width': msg.width,
                        'height': msg.height,
                        'K': list(msg.K),  # [fx,0,cx, 0,fy,cy, 0,0,1]
                    }
    return None


def load_csv_wheel(filepath):
    """加载 Wheel CSV: timestamp_ns,vx,vy,vz,wx,wy,wz"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            ts = int(parts[0])
            vx, vy, vz = float(parts[1]), float(parts[2]), float(parts[3])
            wx, wy, wz = float(parts[4]), float(parts[5]), float(parts[6])
            data.append((ts, vx, vy, vz, wx, wy, wz))
    return data


def wheel_to_joint_velocities(vx, vy, vz, wx, wy, wz):
    """(vx,wz) → 四轮线速度 (m/s), 后轮差速逆运算"""
    v_RL = vx - wz * B_R / 2.0
    v_RR = vx + wz * B_R / 2.0
    return v_RL, v_RR, v_RL, v_RR  # FL, FR, RL, RR (后驱, 前轮=后轮)


def load_undistorted_images(cam_dir):
    """加载去畸变图像，返回按文件名排序的 flattened_1d_numpy_uint8_array 列表"""
    images = []
    png_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    for png_path in png_files:
        img = cv2.imread(png_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray.ravel())  # 1D numpy array (rosbags 要求)
    return images


# ==================== 主流程 ====================

def process_scene(scene_num, src_bag_path, typestore):
    """处理单个 scene"""
    scene_name = f"scene_{scene_num}"
    scene_dir = os.path.join(DATA_DIR, scene_name)
    cam_dir = os.path.join(scene_dir, "cam0", "data")
    wheel_csv = os.path.join(scene_dir, "wheel0", "data.csv")

    # 检查必要文件
    if not os.path.isdir(cam_dir):
        print(f"  [{scene_name}] 跳过: 无 cam0/data/")
        return False
    if not os.path.exists(wheel_csv):
        print(f"  [{scene_name}] 跳过: 无 wheel0/data.csv")
        return False

    # 从原始 bag 读取相机内参
    cam_info = get_camera_info_from_bag(src_bag_path)
    if cam_info is None:
        print(f"  [{scene_name}] 跳过: 无法从 bag 读取 CameraInfo")
        return False

    undistorted_images = load_undistorted_images(cam_dir)
    wheel_data = load_csv_wheel(wheel_csv)

    if not undistorted_images:
        print(f"  [{scene_name}] 跳过: 无去畸变图像")
        return False

    print(f"  [{scene_name}] 去畸变图像: {len(undistorted_images)}, 轮速: {len(wheel_data)}")

    # 读取原始 bag — 使用 header.stamp 绝对时间戳，与轮速天然对齐
    from rosbags.highlevel import AnyReader

    color_images = []      # (ts_ns,)  header.stamp 绝对时间戳
    accel_msgs = []        # (ts_ns, ax, ay, az)
    gyro_msgs = []         # (ts_ns, gx, gy, gz)

    print(f"  [{scene_name}] 读取原始 bag (header.stamp)...")
    with AnyReader([Path(src_bag_path)]) as reader:
        for conn in reader.connections:
            topic = conn.topic
            if topic == '/device_0/sensor_1/Color_0/image/data':
                for _, ts, raw in reader.messages(connections=[conn]):
                    # 图像是 CDR 格式，手动解析 header.stamp
                    sec = struct.unpack_from('<I', raw, 4)[0]
                    nsec = struct.unpack_from('<I', raw, 8)[0]
                    ts_ns = sec * 1_000_000_000 + nsec
                    color_images.append(ts_ns)
            elif topic == '/device_0/sensor_2/Accel_0/imu/data':
                for _, ts, raw in reader.messages(connections=[conn]):
                    msg = reader.deserialize(raw, conn.msgtype)
                    ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    ax = msg.linear_acceleration.x
                    ay = msg.linear_acceleration.y
                    az = msg.linear_acceleration.z
                    accel_msgs.append((ts_ns, ax, ay, az))
            elif topic == '/device_0/sensor_2/Gyro_0/imu/data':
                for _, ts, raw in reader.messages(connections=[conn]):
                    msg = reader.deserialize(raw, conn.msgtype)
                    ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    gx = msg.angular_velocity.x
                    gy = msg.angular_velocity.y
                    gz = msg.angular_velocity.z
                    gyro_msgs.append((ts_ns, gx, gy, gz))

    print(f"  [{scene_name}] 原始: 图像{len(color_images)}, "
          f"Accel{len(accel_msgs)}, Gyro{len(gyro_msgs)}")

    # 合并 IMU: 以 Accel 时间戳为基准, 最近邻匹配 Gyro
    print(f"  [{scene_name}] 合并 IMU...")
    gyro_ts = np.array([g[0] for g in gyro_msgs], dtype=np.int64)
    gyro_vals = np.array([[g[1], g[2], g[3]] for g in gyro_msgs])

    imu_combined = []
    for ts, ax, ay, az in accel_msgs:
        idx = np.searchsorted(gyro_ts, ts)
        if idx >= len(gyro_ts):
            idx = len(gyro_ts) - 1
        elif idx > 0 and abs(gyro_ts[idx] - ts) > abs(gyro_ts[idx-1] - ts):
            idx = idx - 1
        gx, gy, gz = gyro_vals[idx]
        imu_combined.append((ts, gx, gy, gz, ax, ay, az))

    # 获取类型
    Image = typestore.types['sensor_msgs/msg/Image']
    CameraInfo = typestore.types['sensor_msgs/msg/CameraInfo']
    Imu = typestore.types['sensor_msgs/msg/Imu']
    JointState = typestore.types['sensor_msgs/msg/JointState']
    Odometry = typestore.types['nav_msgs/msg/Odometry']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Header = typestore.types['std_msgs/msg/Header']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']
    Point = typestore.types['geometry_msgs/msg/Point']
    Pose = typestore.types['geometry_msgs/msg/Pose']
    PoseWithCovariance = typestore.types['geometry_msgs/msg/PoseWithCovariance']
    Twist = typestore.types['geometry_msgs/msg/Twist']
    TwistWithCovariance = typestore.types['geometry_msgs/msg/TwistWithCovariance']
    ROI = typestore.types['sensor_msgs/msg/RegionOfInterest']

    # 构建消息列表: (topic, serialized_bytes, timestamp_ns)
    messages = []

    # 图像 + CameraInfo (按索引匹配: bag 图像时间戳与去畸变图像文件名时间戳不同, 但顺序一致)
    print(f"  [{scene_name}] 构建图像消息...")
    h, w = cam_info['height'], cam_info['width']
    K = cam_info['K']  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    n_images = min(len(color_images), len(undistorted_images))
    for i in range(n_images):
        ts = color_images[i]
        sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000

        # Image
        img_msg = Image(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                          frame_id='camera_infra1_optical_frame'),
            height=h, width=w,
            encoding='mono8',
            is_bigendian=False,
            step=w,
            data=undistorted_images[i]
        )
        messages.append(('/camera/infra1/image_rect_raw',
                         typestore.serialize_ros1(img_msg, Image.__msgtype__), ts))

        # CameraInfo (D=0 因为图像已去畸变, K 使用原始内参)
        ci_msg = CameraInfo(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                          frame_id='camera_infra1_optical_frame'),
            height=h, width=w,
            distortion_model='plumb_bob',
            D=np.zeros(5, dtype=np.float64),
            K=np.array(K, dtype=np.float64),
            R=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            P=np.array(K[:3] + [0.0] + K[3:6] + [0.0] + K[6:] + [0.0], dtype=np.float64),
            binning_x=0, binning_y=0,
            roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
        )
        messages.append(('/camera/infra1/camera_info',
                         typestore.serialize_ros1(ci_msg, CameraInfo.__msgtype__), ts))

    # IMU
    print(f"  [{scene_name}] 构建 IMU 消息...")
    for ts, gx, gy, gz, ax, ay, az in imu_combined:
        sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000
        imu_msg = Imu(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                          frame_id='camera_imu_optical_frame'),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0),
            orientation_covariance=np.array([-1.0] + [0.0]*8, dtype=np.float64),
            angular_velocity=Vector3(x=gx, y=gy, z=gz),
            angular_velocity_covariance=np.array([-1.0] + [0.0]*8, dtype=np.float64),
            linear_acceleration=Vector3(x=ax, y=ay, z=az),
            linear_acceleration_covariance=np.array([-1.0] + [0.0]*8, dtype=np.float64),
        )
        messages.append(('/camera/imu',
                         typestore.serialize_ros1(imu_msg, Imu.__msgtype__), ts))

    # Wheel → JointState + Odometry (与 ridgeback_light.bag 完全一致)
    print(f"  [{scene_name}] 构建轮速消息...")
    for ts, vx, vy, vz, wx, wy, wz in wheel_data:
        v_FL, v_FR, v_RL, v_RR = wheel_to_joint_velocities(vx, vy, vz, wx, wy, wz)
        sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000

        # JointState
        js_msg = JointState(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec), frame_id=''),
            name=['front_left_wheel', 'front_right_wheel',
                  'rear_left_wheel', 'rear_right_wheel'],
            position=np.zeros(4, dtype=np.float64),
            velocity=np.array([v_FL, v_FR, v_RL, v_RR], dtype=np.float64),
            effort=np.zeros(4, dtype=np.float64),
        )
        messages.append(('/joint_states',
                         typestore.serialize_ros1(js_msg, JointState.__msgtype__), ts))

        # Odometry
        odom_msg = Odometry(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec), frame_id='odom'),
            child_frame_id='base_link',
            pose=PoseWithCovariance(
                pose=Pose(position=Point(x=0.0, y=0.0, z=0.0),
                          orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                covariance=np.array([0.001] + [0.0]*35, dtype=np.float64),
            ),
            twist=TwistWithCovariance(
                twist=Twist(linear=Vector3(x=vx, y=vy, z=vz),
                            angular=Vector3(x=wx, y=wy, z=wz)),
                covariance=np.array([0.001] + [0.0]*35, dtype=np.float64),
            ),
        )
        messages.append(('/ridgeback_velocity_controller/odom',
                         typestore.serialize_ros1(odom_msg, Odometry.__msgtype__), ts))

    # 按时间戳排序
    messages.sort(key=lambda x: x[2])

    # 写入 bag (ROS1 格式, 与 ridgeback_light.bag 一致)
    print(f"  [{scene_name}] 写入 bag ({len(messages)} 条消息)...")
    from rosbags.rosbag1 import Writer

    bag_path = os.path.join(OUTPUT_DIR, f"{scene_name}.bag")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with Writer(bag_path) as writer:
        conn_img = writer.add_connection(
            '/camera/infra1/image_rect_raw', Image.__msgtype__, typestore=typestore)
        conn_ci = writer.add_connection(
            '/camera/infra1/camera_info', CameraInfo.__msgtype__, typestore=typestore)
        conn_imu = writer.add_connection(
            '/camera/imu', Imu.__msgtype__, typestore=typestore)
        conn_js = writer.add_connection(
            '/joint_states', JointState.__msgtype__, typestore=typestore)
        conn_odom = writer.add_connection(
            '/ridgeback_velocity_controller/odom', Odometry.__msgtype__, typestore=typestore)

        topic_to_conn = {
            '/camera/infra1/image_rect_raw': conn_img,
            '/camera/infra1/camera_info': conn_ci,
            '/camera/imu': conn_imu,
            '/joint_states': conn_js,
            '/ridgeback_velocity_controller/odom': conn_odom,
        }

        for topic, raw_cdr, ts in messages:
            conn = topic_to_conn[topic]
            writer.write(conn, ts, raw_cdr)

    print(f"  [{scene_name}] 完成 → {bag_path}")
    return True


def main():
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)

    src_bags = sorted(glob.glob(os.path.join(SRC_BAG_DIR, "*.bag")))
    print(f"找到 {len(src_bags)} 个原始 bag 文件")

    scene_dirs = sorted(
        glob.glob(os.path.join(DATA_DIR, "scene_*")),
        key=lambda x: int(x.split("_")[-1])
    )
    print(f"找到 {len(scene_dirs)} 个 scene 目录\n")

    success = 0
    for scene_dir in scene_dirs:
        scene_num = int(os.path.basename(scene_dir).split("_")[-1])
        if scene_num > len(src_bags):
            print(f"  [scene_{scene_num}] 跳过: 无对应 bag (序号超出)")
            continue
        # 按 scene 编号匹配 bag（1-based → 0-based）
        if process_scene(scene_num, src_bags[scene_num - 1], typestore):
            success += 1

    print(f"\n完成: {success}/{len(scene_dirs)} 个 scene 已处理")


if __name__ == "__main__":
    main()
