#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10.18 数据合成 rosbag (车 IMU + 轮速 + SVC处理后图像 + D455深度图)。

每个 scene 生成一个 bag, 数据全部来自 scene 目录 (不依赖原始 bag):
  - svc_processed/data/*.png   → /camera/infra1/image_rect_raw + /camera/infra1/camera_info
                            时间戳取文件名 (毫秒×1e6 = ns), 灰度 mono8
  - imu1/data.csv         → /camera/imu (车 IMU: timestamp_ns,gx,gy,gz,ax,ay,az)
  - wheel0/data_wz.csv    → /ridgeback_velocity_controller/odom + /joint_states
  - depth0/data.csv+png   → /camera/depth/image_rect_raw (16UC1, 毫米), 仅用于
                            事后深度评测, VIW-Fusion 不订阅

话题名与 config/realsense_d455i/realsense_stereo_imu_config_d455.yaml 一致。
保留 IMU 与轮速共同覆盖的 SVC 图像，覆盖判断使用 SLAM 配置的 td 和 td_wheel。

用法:
  python3 synthesize_bag_svc.py --data-dir /path/to/raw_bags --output-dir /path/to/ros_bags_wz
"""

import argparse
import tempfile
import os
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


def iter_svc_images(img_dir, image_start, image_end):
    """生成器: svc_processed/data/*.png → (ts_ns, topic, serialized_raw)
    文件名即毫秒时间戳×1e6, 图像逐张读取转灰度"""
    typestore = iter_svc_images.typestore
    Image = typestore.types['sensor_msgs/msg/Image']
    CameraInfo = typestore.types['sensor_msgs/msg/CameraInfo']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Header = typestore.types['std_msgs/msg/Header']
    ROI = typestore.types['sensor_msgs/msg/RegionOfInterest']

    K = np.array(SVC_K, dtype=np.float64)
    h, w = SVC_HEIGHT, SVC_WIDTH

    png_files = (path for path in (Path(img_dir) / "data").glob("*.png")
                 if path.stem.isdecimal())
    for png_path in sorted(png_files, key=lambda path: int(path.stem)):
        ts = int(png_path.stem)
        if not image_start <= ts <= image_end:
            continue
        img = cv2.imread(str(png_path))
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


def read_motion_csv(path):
    """Read the same samples for coverage checks and bag serialization."""
    samples = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split(',')
            if fields[0].strip().lower() in {'timestamp', 'timestamp_ns'}:
                continue
            try:
                timestamp = int(fields[0])
                values = tuple(map(float, fields[1:7]))
                if len(values) != 6 or not np.isfinite(values).all():
                    raise ValueError("需要 6 个有限的测量值")
                if timestamp < 0 or (samples and timestamp <= samples[-1][0]):
                    raise ValueError("时间戳必须为非负且严格递增的纳秒整数")
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            samples.append((timestamp, values))
    if len(samples) < 2:
        raise ValueError(f"{path}: 至少需要两条有效样本")
    return samples


def iter_imu(samples):
    """生成器: imu1/data.csv (timestamp_ns,gx,gy,gz,ax,ay,az) → /camera/imu"""
    typestore = iter_svc_images.typestore
    Imu = typestore.types['sensor_msgs/msg/Imu']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Header = typestore.types['std_msgs/msg/Header']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']

    for ts, (gx, gy, gz, ax, ay, az) in samples:
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


def iter_wheel(samples):
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

    for ts, (vx, vy, vz, wx, wy, wz) in samples:
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


def process_scene(scene_dir, typestore, output_dir, td_ns, td_wheel_ns, wheel_name):
    scene_name = os.path.basename(scene_dir)
    img_dir = os.path.join(scene_dir, "svc_processed")
    imu_csv = os.path.join(scene_dir, "imu1", "data.csv")
    wheel_csv = os.path.join(scene_dir, "wheel0", wheel_name)
    depth_dir = os.path.join(scene_dir, "depth0")

    for path, desc in [(img_dir, "svc_processed/"), (imu_csv, "imu1/data.csv"),
                       (wheel_csv, f"wheel0/{wheel_name}")]:
        if not (os.path.isdir(path) if desc.endswith('/') else os.path.exists(path)):
            print(f"  [{scene_name}] 跳过: 缺少 {desc}")
            return False

    bag_path = Path(output_dir) / f"{scene_name}.bag"
    if bag_path.exists():
        raise FileExistsError(f"输出已存在，请使用新目录：{bag_path}")

    imu = read_motion_csv(imu_csv)
    wheel = read_motion_csv(wheel_csv)
    # Estimator: t_imu = t_image + td; t_wheel = t_imu - td_wheel.
    image_start = max(imu[0][0] - td_ns, wheel[0][0] - td_ns + td_wheel_ns)
    image_end = min(imu[-1][0] - td_ns, wheel[-1][0] - td_ns + td_wheel_ns)
    if image_start > image_end:
        raise ValueError("IMU 和轮速没有共同时间覆盖")

    iter_svc_images.typestore = typestore
    streams = [iter_svc_images(img_dir, image_start, image_end), iter_imu(imu),
               iter_wheel(wheel), iter_depth(depth_dir)]
    merged = heapq.merge(*streams, key=lambda x: x[0])

    counts = {t: 0 for t in [TOPIC_IMG, TOPIC_CI, TOPIC_IMU,
                             TOPIC_ODOM, TOPIC_JS, TOPIC_DEPTH]}
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".svc-") as stage:
        temp_bag = Path(stage) / bag_path.name
        with Writer(temp_bag) as writer:
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
        if not counts[TOPIC_IMG]:
            raise ValueError("共同覆盖范围内没有可读取的 SVC 图像")
        os.link(temp_bag, bag_path)

    print(f"  [{scene_name}] 完成 → {bag_path}")
    print(f"    图像: {counts[TOPIC_IMG]}, IMU: {counts[TOPIC_IMU]}, "
          f"轮速: {counts[TOPIC_ODOM]}, 深度: {counts[TOPIC_DEPTH]}")
    return True


def main():
    parser = argparse.ArgumentParser(description="合成经过共同时间覆盖筛选的 SVC ROS1 bag")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]
                        / "config/realsense_d455i/realsense_stereo_imu_config_d455.yaml")
    parser.add_argument("--wheel-csv", choices=("data_wz.csv", "data.csv"), default="data_wz.csv",
                        help="输入 wheel0 下已确认偏航方向的 CSV。脚本不改变 wz")
    args = parser.parse_args()
    config = cv2.FileStorage(str(args.config), cv2.FILE_STORAGE_READ)
    if not config.isOpened():
        parser.error(f"无法读取 SLAM 配置：{args.config}")
    offsets = []
    for key in ("td", "td_wheel"):
        node = config.getNode(key)
        if not (node.isInt() or node.isReal()) or not np.isfinite(node.real()):
            parser.error(f"配置缺少有效的 {key}（单位：秒）")
        offsets.append(round(node.real() * 1e9))
    config.release()

    scenes = sorted((p for p in args.data_dir.glob("scene_*")
                     if p.is_dir() and p.name[6:].isdecimal()), key=lambda p: int(p.name[6:]))
    if not scenes:
        parser.error(f"找不到 scene_N 数据目录：{args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    print(f"输入: {args.data_dir}，输出: {args.output_dir}")
    print(f"时间偏移: td={offsets[0]} ns, td_wheel={offsets[1]} ns")

    success = 0
    for scene in scenes:
        try:
            success += process_scene(str(scene), typestore, args.output_dir, *offsets, args.wheel_csv)
        except (ValueError, OSError) as exc:
            print(f"  [{scene.name}] 处理失败：{exc}")
    print(f"\n完成: {success}/{len(scenes)} 个 scene 已合成 bag")
    return int(success != len(scenes))


if __name__ == "__main__":
    raise SystemExit(main())
