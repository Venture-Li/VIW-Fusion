#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 D455 原始 bag + 去畸变图像 + 轮速 CSV → 合成 ridgeback_light 格式 bag。
[改进] 自动根据时间重叠匹配原始 bag 文件，无需依赖序号对应。
       已手动预置所有 bag 的时间范围，跳过扫描步骤。
"""

import os
import sys
import glob
import struct
import numpy as np
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader

# ==================== 配置 ====================
SRC_BAG_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/8.21/data"
DATA_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/data/静态/"
OUTPUT_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/data/1/"
B_R = 1.672  # 后轮距 (米)
# ==============================================

# =========== 时间范围读取函数（保留，但主函数不再调用） ===========
def get_bag_time_range(bag_path):
    """（保留备用）从 bag 文件中提取首尾时间戳，但主函数已改用预置数据"""
    first_ts = None
    last_ts = None
    try:
        with AnyReader([Path(bag_path)]) as reader:
            for conn, _, rawdata in reader.messages():
                if conn.topic == '/device_0/sensor_1/Color_0/image/data':
                    sec = struct.unpack_from('<I', rawdata, 4)[0]
                    nsec = struct.unpack_from('<I', rawdata, 8)[0]
                    ts = sec * 1_000_000_000 + nsec
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
    except Exception as e:
        print(f"  警告: 读取 {os.path.basename(bag_path)} 失败: {e}")
    return first_ts, last_ts
# =============================================

def get_scene_time_range(scene_dir):
    """从 scene 目录的 wheel0/data.csv 或 PNG 文件名读取首尾时间戳"""
    wheel_csv = os.path.join(scene_dir, "wheel0", "data.csv")
    if os.path.exists(wheel_csv):
        with open(wheel_csv, 'r') as f:
            first_ts = None
            last_ts = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 1:
                    try:
                        ts = int(parts[0])
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                    except ValueError:
                        continue
            if first_ts is not None and last_ts is not None:
                return first_ts, last_ts

    # 备选：从 PNG 文件名读取
    cam_dir = os.path.join(scene_dir, "cam0", "data")
    if os.path.isdir(cam_dir):
        png_files = glob.glob(os.path.join(cam_dir, "*.png"))
        if png_files:
            timestamps = []
            for f in png_files:
                basename = os.path.splitext(os.path.basename(f))[0]
                try:
                    timestamps.append(int(basename))
                except ValueError:
                    continue
            if timestamps:
                return min(timestamps), max(timestamps)
    return None, None

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
                        'K': list(msg.K),
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
    return v_RL, v_RR, v_RL, v_RR

def load_undistorted_images(cam_dir):
    """加载去畸变图像，返回按文件名排序的 flattened_1d_numpy_uint8_array 列表"""
    images = []
    png_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    for png_path in png_files:
        img = cv2.imread(png_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray.ravel())
    return images

# ==================== 核心处理函数 ====================
def process_scene(scene_num, src_bag_path, typestore):
    """处理单个 scene"""
    scene_name = f"scene_{scene_num}"
    scene_dir = os.path.join(DATA_DIR, scene_name)
    cam_dir = os.path.join(scene_dir, "cam0", "data")
    wheel_csv = os.path.join(scene_dir, "wheel0", "data.csv")

    if not os.path.isdir(cam_dir):
        print(f"  [{scene_name}] 跳过: 无 cam0/data/")
        return False
    if not os.path.exists(wheel_csv):
        print(f"  [{scene_name}] 跳过: 无 wheel0/data.csv")
        return False

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

    # 读取原始 bag 的 IMU 和图像时间戳
    color_images = []
    accel_msgs = []
    gyro_msgs = []

    print(f"  [{scene_name}] 读取原始 bag (header.stamp)...")
    with AnyReader([Path(src_bag_path)]) as reader:
        for conn in reader.connections:
            topic = conn.topic
            if topic == '/device_0/sensor_1/Color_0/image/data':
                for _, ts, raw in reader.messages(connections=[conn]):
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

    # 合并 IMU
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

    # 类型定义
    from rosbags.typesys import Stores, get_typestore
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

    messages = []

    # 构建图像消息
    print(f"  [{scene_name}] 构建图像消息...")
    h, w = cam_info['height'], cam_info['width']
    K = cam_info['K']

    n_images = min(len(color_images), len(undistorted_images))
    for i in range(n_images):
        ts = color_images[i]
        sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000

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

        ci_msg = CameraInfo(
            header=Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                          frame_id='camera_infra1_optical_frame'),
            height=h, width=w,
            distortion_model='plumb_bob',
            D=np.zeros(5, dtype=np.float64),
            K=np.array(K, dtype=np.float64),
            R=np.eye(3, dtype=np.float64).flatten(),
            P=np.array(K[:3] + [0.0] + K[3:6] + [0.0] + K[6:] + [0.0], dtype=np.float64),
            binning_x=0, binning_y=0,
            roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
        )
        messages.append(('/camera/infra1/camera_info',
                         typestore.serialize_ros1(ci_msg, CameraInfo.__msgtype__), ts))

    # 构建 IMU 消息
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

    # 构建轮速消息
    print(f"  [{scene_name}] 构建轮速消息...")
    for ts, vx, vy, vz, wx, wy, wz in wheel_data:
        v_FL, v_FR, v_RL, v_RR = wheel_to_joint_velocities(vx, vy, vz, wx, wy, wz)
        sec, nsec = ts // 1_000_000_000, ts % 1_000_000_000

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

    # 写入 bag
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
            writer.write(topic_to_conn[topic], ts, raw_cdr)

    print(f"  [{scene_name}] 完成 → {bag_path}")
    return True

# ==================== 主函数（使用预置 bag 时间范围） ====================
def main():
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # ---------- 预置所有 bag 的时间范围（从你的文本整理） ----------
    bag_time_ranges = {
        "20260821_143538.bag": (1787294138599838734, 1787294150911716223),
        "20260821_143601.bag": (1787294161727842808, 1787294174172991514),
        "20260821_143624.bag": (1787294184088581324, 1787294198001964331),
        "20260821_143648.bag": (1787294208716678381, 1787294223301386118),
        "20260821_143722.bag": (1787294242322573662, 1787294254204690218),
        "20260821_143747.bag": (1787294267019765139, 1787294283840050697),
        "20260821_143817.bag": (1787294297892329216, 1787294309238775492),
        "20260821_143937.bag": (1787294377052859306, 1787294389770249844),
        "20260821_143957.bag": (1787294397513129711, 1787294413129782438),
        "20260821_144027.bag": (1787294427247375488, 1787294442365098238),
        "20260821_144054.bag": (1787294454214134216, 1787294463023549795),
        "20260821_144114.bag": (1787294474206460476, 1787294485952665091),
        "20260821_144134.bag": (1787294494929469585, 1787294507645343781),
        "20260821_144200.bag": (1787294520661604881, 1787294529339027166),
        "20260821_144225.bag": (1787294545523057222, 1787294557239096165),
        "20260821_144249.bag": (1787294569517883778, 1787294580665652275),
        "20260821_144309.bag": (1787294589543040991, 1787294606630241156),
        "20260821_144333.bag": (1787294613705516338, 1787294628289812326),
        "20260821_144432.bag": (1787294672309753418, 1787294682088310003),
        "20260821_144451.bag": (1787294691670041323, 1787294701878335238),
        "20260821_144520.bag": (1787294720334679604, 1787294735621114731),
        "20260821_144545.bag": (1787294745365725040, 1787294757813112020),
        "20260821_144608.bag": (1787294768726293087, 1787294783644553900),
        "20260821_144637.bag": (1787294797694693565, 1787294814985965014),
        "20260821_144713.bag": (1787294833872254610, 1787294854865089893),
        "20260821_144812.bag": (1787294892842864752, 1787294901923087358),
        "20260821_144836.bag": (1787294916104281664, 1787294927686157703),
        "20260821_144857.bag": (1787294937062502623, 1787294959225856543),
        "20260821_144931.bag": (1787294971674295425, 1787294994865393400),
        "20260821_145008.bag": (1787295008485212088, 1787295025637013912),
        "20260821_145038.bag": (1787295038452424526, 1787295057277792215),
        "20260821_145112.bag": (1787295072827103138, 1787295088746612549),
        "20260821_145720.bag": (1787295440304444551, 1787295456724998474),
        "20260821_145748.bag": (1787295468673394203, 1787295485662364006),
        "20260821_145819.bag": (1787295499209658146, 1787295506685387373),
        "20260821_145840.bag": (1787295520136210442, 1787295527477717876),
        "20260821_145854.bag": (1787295534652021170, 1787295547637390852),
        "20260821_145934.bag": (1787295574767529964, 1787295584879197359),
        "20260821_145956.bag": (1787295596360587120, 1787295610012885094),
        "20260821_150023.bag": (1787295623059994221, 1787295637512375832),
        "20260821_150120.bag": (1787295680596853256, 1787295687303997278),
        "20260821_150141.bag": (1787295701521408558, 1787295709198220968),
        "20260821_150833.bag": (1787296113986897707, 1787296125201494455),
        "20260821_150857.bag": (1787296137954200029, 1787296146927611351),
        "20260821_150916.bag": (1787296156172787905, 1787296164619510174),
        "20260821_150935.bag": (1787296175997013807, 1787296184941765547),
        "20260821_150959.bag": (1787296199491250992, 1787296208269277096),
        "20260821_151013.bag": (1787296213710239649, 1787296225655626535),
        "20260821_151052.bag": (1787296252756539583, 1787296260197068930),
        "20260821_151105.bag": (1787296265003896475, 1787296274683407545),
        "20260821_151130.bag": (1787296290234404325, 1787296302315580368),
        "20260821_151150.bag": (1787296310726450682, 1787296321939313412),
        "20260821_151308.bag": (1787296388286270618, 1787296397898377895),
        "20260821_151344.bag": (1787296424898402452, 1787296460940725803),
        "20260821_151700.bag": (1787296620767359257, 1787296626040399551),
        "20260821_151713.bag": (1787296633916618347, 1787296639223173380),
        "20260821_151734.bag": (1787296654309516191, 1787296661149428368),
        "20260821_151801.bag": (1787296681242157221, 1787296688849501133),
        "20260821_151825.bag": (1787296705136006594, 1787296715949283838),
        "20260821_151852.bag": (1787296732070566654, 1787296739881125212),
        "20260821_151918.bag": (1787296758834369183, 1787296766680366039),
        "20260821_151940.bag": (1787296780326881647, 1787296791540530920),
        "20260821_152123.bag": (1787296883050469160, 1787296889425645590),
        "20260821_152146.bag": (1787296906447112322, 1787296913122011662),
        "20260821_152211.bag": (1787296931246834517, 1787296938684883595),
        "20260821_152236.bag": (1787296956675609350, 1787296964783348799),
        "20260821_152259.bag": (1787296979402437449, 1787296987980026960),
        "20260821_152323.bag": (1787297002996442795, 1787297014245806217),
        "20260821_152347.bag": (1787297027192261934, 1787297035535739660),
        "20260821_152407.bag": (1787297047919180870, 1787297058063044786),
        "20260821_152430.bag": (1787297070110911131, 1787297081624922514),
        "20260821_152454.bag": (1787297094807901859, 1787297105854109764),
        "20260821_152517.bag": (1787297117137226105, 1787297125445965052),
        "20260821_152609.bag": (1787297169266603708, 1787297169566896915),
        "20260821_153239.bag": (1787297559769685984, 1787297576425451517),
        "20260821_153304.bag": (1787297584134045601, 1787297597116544724),
        "20260821_153326.bag": (1787297606594466925, 1787297612767136812),
        "20260821_153342.bag": (1787297622214036942, 1787297629721210718),
        "20260821_153358.bag": (1787297638164597750, 1787297648445997238),
        "20260821_153417.bag": (1787297657154321909, 1787297665433492422),
        "20260821_153436.bag": (1787297676344164133, 1787297683018383741),
        "20260821_153454.bag": (1787297694566172123, 1787297702442677259),
        "20260821_153522.bag": (1787297722999929190, 1787297727772135258),
        "20260821_153540.bag": (1787297740121279716, 1787297748465337992),
        "20260821_153602.bag": (1787297762449931145, 1787297773262346745),
        "20260821_153627.bag": (1787297787378089190, 1787297792618604183),
        "20260821_153642.bag": (1787297802766402483, 1787297810206188679),
        "20260821_153700.bag": (1787297820853014469, 1787297825258049488),
    }
    # ----------------------------------------------------------------

    # 构建完整路径字典
    bag_time_ranges_full = {}
    for basename, (start, end) in bag_time_ranges.items():
        full_path = os.path.join(SRC_BAG_DIR, basename)
        bag_time_ranges_full[full_path] = (start, end)

    print(f"已加载 {len(bag_time_ranges_full)} 个 bag 的时间范围（预置）")

    # 获取所有 scene 目录
    scene_dirs = sorted(
        glob.glob(os.path.join(DATA_DIR, "scene_*")),
        key=lambda x: int(x.split("_")[-1])
    )
    print(f"找到 {len(scene_dirs)} 个 scene 目录")

    success = 0
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        scene_num = int(scene_name.split("_")[-1])

        s_start, s_end = get_scene_time_range(scene_dir)
        if s_start is None or s_end is None:
            print(f"  [{scene_name}] 跳过: 无法读取时间范围 (wheel0 或 PNG 缺失)")
            continue

        best_bag = None
        best_overlap = 0
        for bag_path, (b_start, b_end) in bag_time_ranges_full.items():
            overlap = min(s_end, b_end) - max(s_start, b_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_bag = bag_path

        if best_bag is None or best_overlap <= 0:
            print(f"  [{scene_name}] 跳过: 无时间重叠的 bag (scene时间: {s_start}~{s_end})")
            continue

        print(f"  [{scene_name}] 匹配到 {os.path.basename(best_bag)} (重叠时长: {best_overlap/1e9:.1f}s)")
        if process_scene(scene_num, best_bag, typestore):
            success += 1
        else:
            print(f"  [{scene_name}] 处理失败")

    print(f"\n完成: {success}/{len(scene_dirs)} 个 scene 已处理")

if __name__ == "__main__":
    main()