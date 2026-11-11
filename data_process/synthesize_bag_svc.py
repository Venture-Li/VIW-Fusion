#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10.18 数据合成 rosbag (车 IMU + 轮速 + SVC处理后图像 + D455深度图)。

每个 scene 生成一个 bag, 数据全部来自 scene 目录 (不依赖原始 bag):
  - svc_processed/*.png   → /camera/infra1/image_rect_raw + /camera/infra1/camera_info
                            时间戳取文件名 (毫秒×1e6 = ns), 灰度 mono8
  - imu1/data.csv         → /camera/imu (车 IMU: timestamp_ns,gx,gy,gz,ax,ay,az)
  - wheel0/data.csv       → /ridgeback_velocity_controller/odom + /joint_states
  - depth0/data.csv+png   → /camera/depth/image_rect_raw (16UC1, 毫米), 仅用于
                            事后深度评测, VIW-Fusion 不订阅

话题名与 config/realsense_d455i/realsense_stereo_imu_config_d455.yaml 一致。
所有时间戳均为同一墙上时钟域 (ns), 无需时延补偿。

用法:
  python3 synthesize_bag_svc.py
"""

import os
import glob
import heapq
import numpy as np
import cv2
from pathlib import Path
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

# ==================== 配置 ====================
DATA_DIR = "/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/data/raw_bags"
OUTPUT_DIR = "/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/data/ros_bags_wz"
B_R = 1.720  # 后轮距 (米), 10.18 实测 1720mm

# SVC 处理后图像的针孔内参 (= D455 深度内参, 严格深度对齐)
SVC_K = [386.7249, 0.0, 323.3481,
         0.0, 386.7249, 233.6760,
         0.0, 0.0, 1.0]
SVC_WIDTH, SVC_HEIGHT = 640, 480

TOPIC_IMG = '/camera/infra1/image_rect_raw'
TOPIC_CI = '/camera/infra1/camera_info'
TOPIC_IMU = '/camera/imu'
TOPIC_ODOM = '/ridgeback_velocity_controller/odom'
TOPIC_JS = '/joint_states'
TOPIC_DEPTH = '/camera/depth/image_rect_raw'
# ==============================================


def iter_svc_images(img_dir):
    """生成器: svc_processed/*.png → (ts_ns, topic, serialized_raw)
    文件名即毫秒时间戳×1e6, 图像逐张读取转灰度"""
    typestore = iter_svc_images.typestore
    Image = typestore.types['sensor_msgs/msg/Image']
    CameraInfo = typestore.types['sensor_msgs/msg/CameraInfo']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Header = typestore.types['std_msgs/msg/Header']
    ROI = typestore.types['sensor_msgs/msg/RegionOfInterest']

    K = np.array(SVC_K, dtype=np.float64)
    h, w = SVC_HEIGHT, SVC_WIDTH

    png_files = sorted(glob.glob(os.path.join(img_dir, "data", "*.png")))
    for png_path in png_files:
        basename = os.path.splitext(os.path.basename(png_path))[0]
        try:
            ts = int(basename)  # 文件名 = ms×1e6 = ns
        except ValueError:
            continue
        img = cv2.imread(png_path)
        if img is None:
            print(f"    警告: 无法读取 {png_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000

        img_msg = Image(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                          frame_id='camera_infra1_optical_frame'),
            height=h, width=w, encoding='mono8', is_bigendian=False,
            step=w, data=gray.ravel())
        yield ts, TOPIC_IMG, typestore.serialize_ros1(img_msg, Image.__msgtype__)

        ci_msg = CameraInfo(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                          frame_id='camera_infra1_optical_frame'),
            height=h, width=w, distortion_model='plumb_bob',
            D=np.zeros(5, dtype=np.float64), K=K,
            R=np.eye(3, dtype=np.float64).flatten(),
            P=np.array([SVC_K[0], SVC_K[1], SVC_K[2], 0.0,
                        SVC_K[3], SVC_K[4], SVC_K[5], 0.0,
                        SVC_K[6], SVC_K[7], SVC_K[8], 0.0], dtype=np.float64),
            binning_x=0, binning_y=0,
            roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
        yield ts, TOPIC_CI, typestore.serialize_ros1(ci_msg, CameraInfo.__msgtype__)


def iter_imu(imu_csv):
    """生成器: imu1/data.csv (timestamp_ns,gx,gy,gz,ax,ay,az) → /camera/imu"""
    typestore = iter_svc_images.typestore
    Imu = typestore.types['sensor_msgs/msg/Imu']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Header = typestore.types['std_msgs/msg/Header']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']

    with open(imu_csv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                ts = int(parts[0])
                gx, gy, gz = float(parts[1]), float(parts[2]), float(parts[3])
                ax, ay, az = float(parts[4]), float(parts[5]), float(parts[6])
            except ValueError:
                continue
            sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000
            imu_msg = Imu(
                header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                              frame_id='camera_imu_optical_frame'),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0),
                orientation_covariance=np.array([-1.0] + [0.0] * 8, dtype=np.float64),
                angular_velocity=Vector3(x=gx, y=gy, z=gz),
                angular_velocity_covariance=np.array([-1.0] + [0.0] * 8, dtype=np.float64),
                linear_acceleration=Vector3(x=ax, y=ay, z=az),
                linear_acceleration_covariance=np.array([-1.0] + [0.0] * 8, dtype=np.float64))
            yield ts, TOPIC_IMU, typestore.serialize_ros1(imu_msg, Imu.__msgtype__)


def iter_wheel(wheel_csv):
    """生成器: wheel0/data.csv → /ridgeback_velocity_controller/odom + /joint_states"""
    typestore = iter_svc_images.typestore
    Odometry = typestore.types['nav_msgs/msg/Odometry']
    JointState = typestore.types['sensor_msgs/msg/JointState']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Header = typestore.types['std_msgs/msg/Header']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']
    Point = typestore.types['geometry_msgs/msg/Point']
    Pose = typestore.types['geometry_msgs/msg/Pose']
    PoseWithCovariance = typestore.types['geometry_msgs/msg/PoseWithCovariance']
    Twist = typestore.types['geometry_msgs/msg/Twist']
    TwistWithCovariance = typestore.types['geometry_msgs/msg/TwistWithCovariance']

    with open(wheel_csv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                ts = int(parts[0])
                vx, vy, vz = float(parts[1]), float(parts[2]), float(parts[3])
                wx, wy, wz = float(parts[4]), float(parts[5]), float(parts[6])
            except ValueError:
                continue
            sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000

            odom_msg = Odometry(
                header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec), frame_id='odom'),
                child_frame_id='base_link',
                pose=PoseWithCovariance(
                    pose=Pose(position=Point(x=0.0, y=0.0, z=0.0),
                              orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                    covariance=np.array([0.001] + [0.0] * 35, dtype=np.float64)),
                twist=TwistWithCovariance(
                    twist=Twist(linear=Vector3(x=vx, y=vy, z=vz),
                                angular=Vector3(x=wx, y=wy, z=wz)),
                    covariance=np.array([0.001] + [0.0] * 35, dtype=np.float64)))
            yield ts, TOPIC_ODOM, typestore.serialize_ros1(odom_msg, Odometry.__msgtype__)

            # joint_states: (vx,wz) → 四轮线速度 (后轮差速逆运算)
            v_RL = vx - wz * B_R / 2.0
            v_RR = vx + wz * B_R / 2.0
            js_msg = JointState(
                header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec), frame_id=''),
                name=['front_left_wheel', 'front_right_wheel',
                      'rear_left_wheel', 'rear_right_wheel'],
                position=np.zeros(4, dtype=np.float64),
                velocity=np.array([v_RL, v_RR, v_RL, v_RR], dtype=np.float64),
                effort=np.zeros(4, dtype=np.float64))
            yield ts, TOPIC_JS, typestore.serialize_ros1(js_msg, JointState.__msgtype__)


def iter_depth(depth_dir):
    """生成器: depth0/data.csv+png → /camera/depth/image_rect_raw (16UC1, 毫米)"""
    typestore = iter_svc_images.typestore
    Image = typestore.types['sensor_msgs/msg/Image']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Header = typestore.types['std_msgs/msg/Header']

    csv_path = os.path.join(depth_dir, "data.csv")
    if not os.path.exists(csv_path):
        return
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            png_path = os.path.join(depth_dir, "data", parts[1].strip())
            depth = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f"    警告: 无法读取深度图 {png_path}")
                continue
            h, w = depth.shape[:2]
            sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000
            img_msg = Image(
                header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                              frame_id='camera_depth_optical_frame'),
                height=h, width=w, encoding='16UC1', is_bigendian=False,
                step=w * 2,
                data=depth.astype(np.uint16).ravel().view(np.uint8))
            yield ts, TOPIC_DEPTH, typestore.serialize_ros1(img_msg, Image.__msgtype__)


def process_scene(scene_dir, typestore):
    scene_name = os.path.basename(scene_dir)
    img_dir = os.path.join(scene_dir, "svc_processed")
    imu_csv = os.path.join(scene_dir, "imu1", "data.csv")
    wheel_csv = os.path.join(scene_dir, "wheel0", "data_wz.csv")
    depth_dir = os.path.join(scene_dir, "depth0")

    for path, desc in [(img_dir, "svc_processed/"), (imu_csv, "imu1/data.csv"),
                       (wheel_csv, "wheel0/data_wz.csv")]:
        if not (os.path.isdir(path) if desc.endswith('/') else os.path.exists(path)):
            print(f"  [{scene_name}] 跳过: 缺少 {desc}")
            return False

    bag_path = os.path.join(OUTPUT_DIR, f"{scene_name}.bag")
    if os.path.exists(bag_path):
        os.remove(bag_path)

    iter_svc_images.typestore = typestore

    # 四路生成器各自按时间有序, 用 heapq.merge 全局按时间归并, 避免整包驻留内存
    streams = [iter_svc_images(img_dir), iter_imu(imu_csv),
               iter_wheel(wheel_csv), iter_depth(depth_dir)]
    merged = heapq.merge(*streams, key=lambda x: x[0])

    counts = {t: 0 for t in [TOPIC_IMG, TOPIC_CI, TOPIC_IMU,
                             TOPIC_ODOM, TOPIC_JS, TOPIC_DEPTH]}
    with Writer(bag_path) as writer:
        conns = {t: writer.add_connection(t, msgtype, typestore=typestore)
                 for t, msgtype in [
                     (TOPIC_IMG, 'sensor_msgs/msg/Image'),
                     (TOPIC_CI, 'sensor_msgs/msg/CameraInfo'),
                     (TOPIC_IMU, 'sensor_msgs/msg/Imu'),
                     (TOPIC_ODOM, 'nav_msgs/msg/Odometry'),
                     (TOPIC_JS, 'sensor_msgs/msg/JointState'),
                     (TOPIC_DEPTH, 'sensor_msgs/msg/Image')]}
        for ts, topic, raw in merged:
            writer.write(conns[topic], ts, raw)
            counts[topic] += 1

    print(f"  [{scene_name}] 完成 → {bag_path}")
    print(f"    图像: {counts[TOPIC_IMG]}, IMU: {counts[TOPIC_IMU]}, "
          f"轮速: {counts[TOPIC_ODOM]}, 深度: {counts[TOPIC_DEPTH]}")
    return True


def main():
    typestore = get_typestore(Stores.ROS1_NOETIC)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scene_dirs = sorted(
        glob.glob(os.path.join(DATA_DIR, "scene_*")),
        key=lambda x: int(x.split("_")[-1]))
    print(f"找到 {len(scene_dirs)} 个 scene 目录")
    print(f"输出目录: {OUTPUT_DIR}")

    success = 0
    for scene_dir in scene_dirs:
        if process_scene(scene_dir, typestore):
            success += 1
        else:
            print(f"  [{os.path.basename(scene_dir)}] 处理失败")

    print(f"\n完成: {success}/{len(scene_dirs)} 个 scene 已合成 bag")


if __name__ == "__main__":
    main()
