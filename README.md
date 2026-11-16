# VIW-Fusion 车辆减速实验

本项目在车辆从 20 或 30 km/h 减速至停车的过程中，用单目 SVC 图像、车载 IMU 和轮速计估计轨迹与稀疏地图点。评测时，将选定帧的地图点与配准到 SVC 图像的 D455 深度比较，计算深度误差并检查尺度偏差。D455 深度用于离线评测。

## 主要改动

| 问题 | 改动 |
| --- | --- |
| 轮速偏航方向反了 | 对本批数据的 `wheel0/data.csv` 只取反偏航角速度 `wz`，写入 `data_wz.csv`，再生成 `ros_bags_wz`。时间戳和线速度保持原样。 |
| 序列首尾缺少融合数据 | 按 SVC、IMU 和轮速的共同时间覆盖裁剪首尾图像，计算覆盖时考虑配置中的时间偏移。合包时自动筛选图像，并完整保留 IMU 和轮速样本。 |
| bag 播完时后端仍在处理 | 播放前统计图像总数并传给 `viwo_node`。前端处理完预期图像后通知后端收尾，脚本通过 `wait` 确认节点成功退出，再停止录制并归档结果。 |
| 初始化失败或重试过多 | 将初始化 BA 代价阈值从 `0.02` 放宽到 `0.03`，补充 PnP 和 BA 诊断日志。对照实验中，初始化成功率从 19/20 提升到 20/20，`scene_6` 从第 108 次尝试缩短到第 1 次，`scene_9` 从 60 次未成功变为第 6 次成功。 |
| 停车漂移、地图点不足及尺度偏大 | 调整 IMU 噪声参数为 `acc_n=0.005`、`gyr_n=0.0005`、`acc_w=0.0002`、`gyr_w=0.000018`。调整后，`scene_15` 恢复了可评测地图点。轮速纵向比例 `sx=0.95` 是本批数据的临时尺度修正。 |
| 序列中间出现采样缺口 | `scene_18` 的 IMU 和轮速存在约 0.4 秒的采样缺口。帧间仅取得一条边界样本时，预积分用相邻图像的时间差作为步长，以现有读数近似运动。缺口原因仍待排查。 |
| D455 深度与 SVC 边缘错位 | 匹配前给 D455 时间戳加 `0.57 s`，再用外参将深度配准到 SVC。偏移量适用于本批数据，换数据时应重新测量。 |

## 1. 准备数据

已有修正并检查过时间覆盖的 `ros_bags_wz/scene_*.bag` 时，可直接用于运行。离线评测还需要原始数据目录下每个场景的 SVC 图像和 D455 深度。目录示例：

```text
data/
├── ros_bags_wz/
│   └── scene_1.bag
└── raw_bags/
    └── scene_1/
        ├── svc_processed/data/   # 已处理的 SVC PNG，文件名为纳秒时间戳
        ├── imu1/data.csv
        ├── wheel0/data.csv
        └── depth0/              # D455 深度图及时间戳索引
```

从原始数据重建时，在宿主机的 Python 3 环境安装 `numpy`、`opencv-python`、`rosbags`，然后从仓库根目录运行：

```bash
python3 data_process/synthesize_bag_svc.py \
  --data-dir /path/to/data/raw_bags \
  --output-dir /path/to/data/ros_bags_wz \
  --config config/realsense_d455i/realsense_stereo_imu_config_d455.yaml
```

脚本会处理输入目录下所有 `scene_N`，从 SLAM 配置读取 `td` 和 `td_wheel`，只写入 IMU 和轮速均覆盖的 SVC 图像及对应的相机信息。IMU、轮速样本完整保留。脚本不会覆盖已有的同名 bag。这个检查针对序列首尾，序列中间的采样缺口需要另行检查。

默认读取 `wheel0/data_wz.csv`。如果同事的 `wheel0/data.csv` 已确认偏航方向正确，可加 `--wheel-csv data.csv` 直接使用。合包脚本不会修改 `wz`。旧数据的取反流程见 [lwc_wheel_wz.py](data_process/lwc_wheel_wz.py)，该脚本按旧批次的目录和场景编号运行。

## 2. 编译并运行 SLAM

使用 Docker 中的 ROS Melodic、Ceres 1.14 和 Sophus。在仓库根目录执行以下命令，将 `DATA_DIR` 改成包含 `ros_bags_wz` 的宿主机目录：

```bash
DATA_DIR=/absolute/path/to/data
docker build -t ros:vins-fusion docker
docker run -d --name viw-dev \
  -v "$PWD:/root/catkin_ws/src/VIW-Fusion" \
  -v "$DATA_DIR:/data:ro" \
  ros:vins-fusion sleep infinity

docker exec viw-dev bash -lc 'source /opt/ros/melodic/setup.bash && cd /root/catkin_ws && catkin_make'
```

已有同名容器时，确认挂载路径一致，执行 `docker start viw-dev`，并在代码更新后重新编译。当前 [run_all.sh](run/run_all.sh) 使用固定路径，默认读取容器内 `/data/ros_bags_wz/scene_1.bag` 至 `scene_20.bag`，找不到的序列会跳过。如果脚本仍指向 `/data/ros_bags` 或只循环一个序列，请把 `BAG_DIR` 改为 `/data/ros_bags_wz`，并将循环范围改为要运行的场景编号（全量为 `seq 1 20`）。然后在宿主机运行：

```bash
VISUALIZE=0 bash run/run_all.sh
```

如需录制特征跟踪图，先将脚本中的 `HOST_OUTPUT` 改为本仓库的 `output` 绝对路径，将 `HOST_DATA_BASE` 改为跟踪 bag 的归档目录，再用 `VISUALIZE=1` 运行。主节点日志、轨迹、最终窗口位姿和稀疏点分别写入 `output/scene_N/log/` 下的 `viwo_node.log`、`vio.csv`、`pose_output.txt` 和 `points_output.txt`。节点正常退出表示处理收尾完成，评测仍需要有效的位姿和地图点文件。

## 3. 对比 D455 深度

在宿主机的 Python 3 环境安装评测依赖 `numpy`、`opencv-python`、`matplotlib`，然后从仓库根目录运行：

```bash
DATA_ROOT_BASE="$DATA_DIR/raw_bags" bash eval/batch_eval_svc.sh \
  --calibration eval/depth_offset.json \
  --frame-index 8
```

`--frame-index` 是 `pose_output.txt` 的行索引，从 0 开始。这里的 8 选第九行，用于复现时间对齐检查。批量脚本默认读取本仓库的 `output` 和 [D455 实验配置](config/realsense_d455i/realsense_stereo_imu_config_d455.yaml)，可通过 `OUTPUT_DIR` 和 `CONFIG` 指向实际运行时使用的目录和配置。评测默认允许匹配时间差为 50 ms，可用 `--max-time-diff` 收紧。换用其他设备或开启在线相机外参估计时，参见 [标定与评测说明](eval/README_svc_eval.md)。

| 文件 | 查看什么 |
| --- | --- |
| `output/summary_svc.csv` | 各序列评测是否成功 |
| `output/scene_N/depth_eval_result.json` | 有效点数、相对深度误差、尺度斜率、匹配时间差 |
| `output/scene_N/depth_comparison.csv` | 逐点估计深度和实测深度 |
| `output/scene_N/color_depth_overlay.png` | D455 深度轮廓与 SVC 图像的对齐情况 |
| `output/scene_N/depth_scatter.png` | 估计深度与实测深度的散点图，尺度斜率越接近 1，整体比例越接近 |

深度误差反映所选帧的稀疏点深度质量。评测中的 `PASS` 表示成功完成计算，精度请查看对应的误差指标。
