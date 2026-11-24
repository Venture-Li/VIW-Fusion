# VIW-Fusion 新数据实验交接（2026-11-23）

这份文档给接手的新 Codex 使用。用户希望说明简洁、通俗；做实验时先核对数据和已有文件，不要把旧批次的标定、时间偏移和轮速符号直接套到新批次。用户此前多次明确要求先做隔离实验、找到原因，再决定是否改正式源码；不要擅自覆盖正式 `output` 或还原工作区已有改动。

## 1. 项目做什么、当前状态

- 输入：车辆减速至停车期间的单目 SVC 图像、车载 IMU、轮速计。SLAM 输出轨迹和滑窗稀疏地图；D455 深度**只用于离线评测**，不参与 SLAM。
- 主配置：`config/realsense_d455i/realsense_stereo_imu_config_d455.yaml`；主程序在 Docker 容器 `viw-dev` 内编译、运行；批量入口 `run/run_all.sh`；评测入口 `eval/batch_eval_svc.sh` / `eval/eval_depth_svc.py`。先读仓库根目录 `README.md` 和 `eval/README_svc_eval.md`，本文件补充了旧批次排障证据与新数据注意事项。
- 当前配置（务必以运行时文件为准）：`max_cnt=300`、`enable_timing=1`、`estimate_wheel_intrinsic=0`（固定轮速内参）、`sx=0.95, sy=1, sw=1`、`estimate_wheel_extrinsic=0`、`estimate_td=0, td=-0.048 s`、`estimate_td_wheel=0, td_wheel=0`。IMU 参数为 `acc_n=0.005, gyr_n=0.0005, acc_w=0.0002, gyr_w=0.000018`。`sx=0.95` 是旧批次的临时尺度校正，新数据需要重新核验。
- `estimator.cpp` 中地面旋转打包错误已由用户手工修复：`Quaterniond q2{rpw};` 后 `para_plane_R[0][0..3]` 现在写的是 `q2.x/y/z/w()`。检查此处，不要误改轮速外参的 `q`。
- 仓库工作区**不是干净状态**，包含历史代码、配置、README 和大量 `output` 改动；不要 `git reset`、清理或覆盖用户成果。运行前用 `git status --short` 和配置文件核对当下状态。

## 2. 已查明的问题与处理

| 问题 | 已有证据、处理及边界 |
| --- | --- |
| 旧批次轮速偏航角速度 `wz` 符号反 | 与车载 IMU 对照发现方向相反；`data_process/lwc_wheel_wz.py` 将旧批次 `wheel0/data.csv` 的 `wz` 取反写为 `data_wz.csv`，其他列和时间戳不变。**新数据先验符号，再决定是否取反**；不能因为旧批次反了就自动反。`fix_bag_wheel_wz.py` 仅是针对特定旧 bag 的对照脚本，不是标准合包步骤。 |
| 序列首尾 SVC 超出 IMU/轮速共同覆盖 | `data_process/synthesize_bag_svc.py` 读取 YAML 的 `td`、`td_wheel`，只合入同时被两种传感器覆盖的 SVC 帧，IMU/轮速原始样本保留。它解决**首尾覆盖**，不自动填补序列中间的缺口；旧 `scene_18` 中间约有 0.4 s 采样缺口，后端预积分已有单边界样本近似处理，缺口来源未查明。 |
| bag 播放结束后输出不完整 | `run/run_all.sh` 先统计 bag 图像数，传给 `viwo_node`；前后端完成并正常退出后脚本才归档。`rosbag play` 结束不等于算法处理结束。 |
| 初始化失败/过慢 | 初始化 BA 代价阈值由 `0.02` 放宽为 `0.03`；旧批次先前从 19/20 提高到 20/20。旧批次 IMU 参数调整后停车漂移和地图点不足有所改善。新批次初始化仍须看新日志，不能假设 20/20。 |
| 开启轮速内参估计反而变差 | 旧数据 `vy` 全程为 0，`sy` 不可观测，曾漂到百万级；即使固定 `sy`、只估 `sx/sw`，整体误差也未稳定改善。当前选择 `estimate_wheel_intrinsic=0` 固定三项。优化器降低自身残差，不保证离线 D455 深度误差变小。新数据若要重启内参估计，须单独对照验证可观测性和精度。 |
| `max_cnt` 变大反而部分序列无地图 | 先前以为是停车后频繁换关键帧、旧点滑出窗口。深入实验表明这是连锁反应：`estimator.cpp` 原来把地面旋转 `rpw` 打包成轮速外参旋转 `q`，每次优化/边缘化前覆盖地面方向；`scene_16` 在未停稳时就开始掉点。临时只修正该传值，旧 `scene_16`、400 点从最终 0 个点恢复到 366 个点，最终滑窗 1/3/5 帧平均相对误差 2.40%，最后 80 个后端帧的关键帧数由 80 降到 2。用户随后已将这四行修进正式源码。新数据仍可能有独立的特征或标定问题，不可把所有失败归于此处。 |
| D455 深度与 SVC 轮廓重影 | 旧批次 D455 和车端时钟不同；`eval/depth_offset.json` 为 `{"depth_time_offset_s": 0.57}`，评测配对时给 D455 文件时间戳 **+0.57 s**。只影响评测，不改 SLAM。**新批次必须重新估计该偏移**，并核对 SVC↔D455 外参、内参、深度单位；不能直接沿用旧 JSON。 |

还有一个未正式修改的特征生命周期风险：`feature_manager.cpp` 三角化要求点至少两次观测且 `start_frame<=WINDOW_SIZE-2`，BA 部分只按至少两次观测纳入，未有有效深度的年轻点可能进入优化。曾在临时探针里尝试“放宽三角化”或“给待估点正深度初值”，虽然留下了点，最终窗口 1/3/5 帧却没有有效 D455 对应，故**不要把这些实验补丁合入**。修复地面旋转后旧批次已能 20/20 评测。

## 3. 旧批次的精度/耗时基线（仅供对照）

在旧批次 20 个 `scene` 上，评测取**最后滑窗** `--frame-index 1、3、5`，每个序列先对三帧的指标取算术平均，再对 20 序列等权平均。此前隔离实验比较四档：

| `max_cnt` | 方案 | 初始化/三帧可评测 | 20 序列平均相对误差 | 最差序列平均相对误差 | 后端单帧平均总耗时 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 150 | 临时修正地面旋转 | 20/20 | 3.68% | `scene_4` 9.79% | 35.9 ms |
| 250 | 临时修正地面旋转 | 20/20 | 3.70% | `scene_17` 6.47% | 57.9 ms |
| **300** | **临时修正地面旋转** | **20/20** | **3.43%** | **`scene_19` 5.23%** | **69.7 ms** |
| 600 | 原逻辑旧基准 | 20/20 | 3.30% | `scene_17` 7.70% | 144.1 ms |

因此旧批次推荐 300 点：相比 600 点大致减半后端单帧耗时，平均误差接近，同时避免 150/250 点个别序列明显恶化。**这不证明 300 是数学最小值，也不保证新批次最优。** 点数改变会改变选中点、初始化时机、关键帧和优化约束；成功退出、有点云不等于精度合格。

用户后来正式手工修复源码并以 `max_cnt=300` 跑完整批旧数据。正式运行 20/20 初始化，最后窗口 1/3/5 帧共 60/60 次评测成功，20 序列等权平均相对误差约 **3.424%**。正式结果中 `scene_16` 三帧平均相对误差约 **3.30%**，`scene_14` 约 **1.80%**；与上述隔离实验略有浮动属正常。耗时以各场景 `output/scene_N/log/timing.csv` 为准，筛选 `phase=steady`；上表的 69.7 ms 来自隔离实验，正式运行各序列约 56–83 ms。旧 `output` 可能随后被重跑覆盖，不能仅按文件名认定是哪次实验；初始化日志可能同时留有多次运行文件，按时间选择本次文件，勿把旧、新日志尝试次数相加。

## 4. 时间与标定：新批次必须核对

1. **SVC ↔ 车载 IMU**：当前 `td=-0.048 s`，SLAM 使用 `t_imu=t_image+td`，即旧批次图像时间减 48 ms 与 IMU 对齐；当前固定，不在线估计。新批次重新核验。
2. **轮速 ↔ IMU**：当前 `td_wheel=0`，SLAM 对轮速不额外平移。旧数据提取脚本 `data_process/extract_wheel_processtd.py` 曾给轮速 CSV 时间戳 **+99 ms**；这属于数据制作阶段，只有确认新数据沿用同一时钟处理时才可沿用。不可在 CSV 已平移后又给 `td_wheel` 加同样补偿。
3. **D455 深度 ↔ SVC**：旧 `depth_time_offset_s=+0.57 s`，只在离线评测时使用。`eval/eval_depth_svc.py` 还支持 `svc_time_offset_s`（默认 0）；文件时间戳加 offset 后匹配目标 pose 时间。新设备/时钟/导出流程要重新测量。
4. 评测的 `--max-time-diff` 默认 0.05 s 是**匹配容差**，不是额外时间偏移。`eval/depth_offset.json` 不参与合包或 SLAM。

## 5. 新数据上手流程

先确认新数据目录结构、scene 编号、图片是否已完成 `svc_processed` 处理、IMU/轮速 CSV 列与时间戳单位、D455 的 `depth0` 格式，以及容器 `/data` 挂载。旧数据宿主机根目录是 `/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/data`；以下命令中的 `/path/to/data` 要换成实际**新数据**目录。**现有 `run_all.sh` 写死了容器名 `viw-dev`、`/data/ros_bags_wz`、序列 `1..20` 和旧机器的跟踪 bag 归档路径；新数据的序列数/挂载不同要先改脚本或准备隔离副本。**不要在不确认的情况下重用旧输出目录。

### A. 准备轮速与合 bag

- 判断新批次 `wz` 符号：把轮速角速度转到 IMU 坐标系、时间对齐后比较明显转弯段；`vx` 也要检查。`lwc_wheel_wz.py` 写死旧路径和 1–20 序列，不能原样用于新目录。若新 CSV 符号正确，合 bag 用 `--wheel-csv data.csv`；若确实反了，先对新目录生成只反 `wz` 的 `data_wz.csv`，再用默认输入。
- 合 bag（宿主机 Python，需 `numpy/opencv-python/rosbags`）：

```bash
python3 data_process/synthesize_bag_svc.py \
  --data-dir /path/to/data/raw_bags \
  --output-dir /path/to/data/ros_bags_wz \
  --config config/realsense_d455i/realsense_stereo_imu_config_d455.yaml \
  --wheel-csv data.csv  # 仅当确认原 CSV 的 wz 已正确
```

脚本遇到已有同名 bag 会拒绝覆盖。合包会检查首尾共同覆盖；内部缺采样另查。`data_process/fix_bag_wheel_wz.py` 不是默认步骤。

### B. 编译、运行

确保 Docker 的 `/data` 绑定到**新数据目录**，不然 `docker start viw-dev` 可能因旧外置盘路径不存在而失败。已正确挂载时，在宿主机运行：

```bash
docker start viw-dev
docker exec viw-dev bash -lc \
  'source /opt/ros/melodic/setup.bash && cd /root/catkin_ws && catkin_make -j2'
VISUALIZE=0 bash run/run_all.sh
```

`VISUALIZE=0` 只关闭 `image_track` 录制，不关闭 SLAM。若需录制、当前路径已正确设置，则可用默认 `bash run/run_all.sh`。确认每个序列日志包含正常收尾、输入帧数正确，并查看 `output/scene_N/log/pose_output.txt`、`points_output.txt` 是否非空。

### C. 深度评测与计时

先为新批次确认 D455/SVC 标定和时间偏移，准备新的 calibration JSON；**不要先用旧 `eval/depth_offset.json` 产生看似精确的数字**。单帧批量评测示例：

```bash
DATA_ROOT_BASE=/path/to/data/raw_bags bash eval/batch_eval_svc.sh \
  --calibration /path/to/new_calibration.json --frame-index 1
```

对 3、5 分别再运行一次。注意批量脚本每次会覆盖同一场景的 `depth_eval_result.json`、`depth_comparison.csv`、图片及 `output/summary_svc.csv`；若要三帧平均，**每次运行后先归档结果**，或在临时场景目录分别评测，不能只看最后一次。`--frame-index` 是最终 `pose_output.txt` 从 0 开始的行索引。评测 PASS 只代表能算出对应点，仍需看有效点数、相对误差、散点斜率和配准叠图。`No valid depth correspondences` 可能是最终点云空/当前帧未观测、时间/标定不匹配、深度筛选后无点，先逐项检查，不要直接调阈值掩盖问题。

`timing.csv` 包含 `feature_tracking_ms`、两种预积分、`problem_build_ms`、`ceres_solve_ms`、`marginalization_ms`、`near_refinement_ms`、`frame_total_ms`。统计稳态后端时筛选 `phase=steady`；前端跟踪可对所有输入帧单独平均。`frame_total_ms` 是同一张图的计算耗时之和，不含 bag 播放、队列等待等墙钟时间。P95 是第 95 百分位耗时。

## 6. 接手时的优先判断

先核对**输入数据、时间轴、轮速方向与标定**，再看初始化日志和地图点生命周期，最后比较精度/耗时。不要把旧批次的 `sx=0.95`、`td=-0.048`、轮速 +99 ms、D455 +0.57 s 当成新批次真值。新数据实验若需改源码，先在独立位置或隔离输出做单因素对照，报告“运行是否完成、可评测点数、1/3/5 帧误差、耗时”，再决定正式改动。


---

## 2026-11-23 接手后的20/30 km/h验证结论

- 新30原始44个bag：原版24/44初始化。已确认新svc_processed相对旧投影有固定10.828727度旋转，旧外参不能直接解释该图像坐标；统一到旧20可信投影后，原版提高到32/44。
- 原片段42/44末尾轮速仍非零。完整车采原始记录还有被截掉的真实停车尾段；保留原高速段、统一投影、按真实三传感器覆盖补回尾段后，**原版程序44/44初始化、44/44非空地图**，全部图像处理完毕，最终各11个位姿。SfM接力实验版也44/44，当前不需要为此改正式算法。
- 旧20同机原版复跑：60/60评测成功，平均相对误差3.420267%，历史原结果复评3.423943%。两组算法/标定参数一致，仅结果保存路径不同；本次没有替换正式算法、配置、原始bag或旧output。
- 最终数据：outputs/secone_init_diagnosis/restored_bags/；推荐结果：outputs/secone_init_diagnosis/validation/restored_original30/；原版运行时快照：outputs/secone_init_diagnosis/original_runtime/。
- 证据与完整说明：outputs/viw_bootstrap_validation_20261123/experiment_report.md、final_summary.json；输入审计：outputs/secone_init_diagnosis/validation/restored30/input_audit.json，44/44通过，图像越界0。新增内容均来自原始真实记录，未补造零速度/IMU、未填补scene33已有约0.6秒缺口。
- 其余大量算法实验没有通过旧精度回归，不要把outputs下的原型复制进正式源码。最小SfM接力版旧平均3.233%但个别场景变差，只保留为可选实验。
- 新30绝对深度精度仍未验收：D455的新外参、时钟偏移及有效重叠范围未确认；不得沿用旧+0.57秒，亦不得采用实验搜索的1.075秒或单场景停稳事件粗估作定论。50数据尚未运行。

### 2026-11-24 更正与原片段复核

用户明确要求：按原片段已有数据终点计算，不追加停车尾段；彻底找到早期初始化问题前，不改正式源码。上节的44/44仅证明补入真实尾段后**最终**可初始化，其中六个原失败场景都是在原片段结束之后才首次进入稳态，不能视作高速段初始化解决，也不再作为当前推荐方案。

用户同事提供的新虚拟相机外参旋转与图像几何实测相差约0.0052度；相机内参和项目现有svc_processed.yaml一致。对原始44个30km/h bag隔离复跑同事配置（含td=-0.056，场景33同配置单独重跑排除偶发播放超时），44/44输入处理完，**21/44初始化**，成功编号与此前修正外参但td=-0.048的对照完全相同。23个失败场景764次尝试的门槛/阶段分布与证据见 `outputs/secone_init_diagnosis/validation/teammate_original30/diagnosis.md`。正式源码和正式配置未改。

轮速符号：新批次原始`wheel0/data.csv`的wz仍与IMU反向；44个bag共27029条轮速消息逐条对应已反号的`data_wz.csv`，bag内方向正确。不要再把bag内轮速符号当作当前初始化失败原因，也不要把原始CSV误称为已经正确。

阈值与本质原因追加分析：23个失败场景764次初始化尝试中，最早四分之一尝试171/200次死在视觉SfM；末四分之一才更多出现IMU激励/BA门槛。PnP对应点不足10的263次里，即使降到6仍有127次不足6；104次外层BA超0.03的中位代价0.152，放宽到0.06仅18次尝试会穿过该门槛，且不能保证正确。191次SfM内部BA失败均达到50次迭代上限，代价往往远高于低代价通道。旧20成功批次也有334次IMU激励拒绝，故单次阈值拒绝不是新30的独有根因。固定参考帧/末帧建图导致短寿命特征不能顺利接力，轮速只在SfM之后的VisualIMUAlignment使用；前端多线程下无后端点预测且初始化期几何F剔除未启用。离线图像跟踪抽样支持新视角部分场景轨迹更短，但非VIO内部真实特征统计。详见`outputs/secone_init_diagnosis/validation/teammate_original30/threshold_vs_geometry.md`。仍未有精度验收合格的早初始化算法，不改正式源码。

再核对矛盾（2026-11-24）：完整44段首约1.5秒图像审计表明，失败场景相邻帧二维匹配中位数并不比成功场景少（LK25 vs18.5），不能笼统说高速图像“没特征”；多帧可信3D与二维点数是不同问题。使用同事外参、IMU、已修正轮速及跨三帧正深度/视差>1°/三帧重投影<2px检查，ORB300点预算下失败场景三帧有效候选中位8，23段中10段中位>=10；ORB1500预算下22/23段中位>=10，不能把1500结果冒充正式300点前端。scene20在300预算下中位约34.5个可信三帧点，但原版32次SfM全失败；旧外参负对照降至5，证明匹配外参的重要性。scene31在300预算下只有2个，虽二维相邻匹配多。这表明一条流程中有取点/跟踪、短轨迹转初始地图、提前初始化后持续精度三类问题。旧20 scene11只改初始SfM，首输出从15.363s提前到0.699s，但最终深度2.573%退至3.888%，因此既有候选未通过用户的精度约束。详见`outputs/secone_init_diagnosis/validation/teammate_original30/root_cause_audit_20261124.md`；正式源码仍未改，30任务未解决。

---

## 2026-11-24 授权后隔离试验交接（用户要求停止，供下一位 agent 接手）

**当前状态：已停止全部本轮运行。** `viw-diag-fullrate` 容器已停止（`exited`），最后一次 `ablate_skip_visual_margin20/scene_17` 已正常跑完并归档；没有继续跑 44 段或 50 km/h。正式 `vins_estimator/src/`、正式配置及原始数据**本轮未改**；本轮新增/修改的 C++ 仅在 `/home/nio/code/VIW-Fusion/outputs/secone_init_diagnosis/source/` 隔离副本，**不要直接复制到正式源码**。正式 Git 工作区在接手前已有许多脏文件，勿清理/覆盖。用户已授权隔离实验，但要求彻底定位、20 km/h 精度不得下降，现又明确要求停止并交接。

### 路径与输入

- 项目根：`/home/nio/code/VIW-Fusion`。
- 30 km/h **原始长度** 44 bag：`/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/data/secone/bags_svc_wall056/`，对应原始展开数据 `.../data/secone/data/scene_N/`；没有补停车尾段。30 的同事配置：`outputs/secone_init_diagnosis/validation/teammate_original30/config.yaml`；原版正确视角/外参/td=-0.056 对照 **21/44**，详见同目录 `diagnosis.md`、`root_cause_audit_20261124.md`、`root_fix_design_review.md`。新 bag 的轮速角速度符号已经修正；原 CSV 未修正。
- 20 km/h bag：`.../data/one/ros_bags_wz/`，D455 评测原始数据：`.../data/one/raw_bags/`。注意旧20原 CSV 的轮速 wz 与 IMU 相反，但 `ros_bags_wz/scene_17.bag` 消息已翻正（bag 消息首值 +0.072674，原 CSV -0.072674），所以此次旧20回退不是使用了错误轮速符号。
- 所有本轮模式及逐场景 `run_batch.sh`、`config.yaml`、`scene_N/log/`：`outputs/secone_init_diagnosis/validation/ablate_*/` 和 `trace_*/`。旧20评测按最终 `pose_output.txt` 的第 1、3、5 行，独立归档于 `outputs/viw_bootstrap_validation_20261123/eval_模式名/results.json`；**必须同时看 `valid_points`，不能只看误差百分比**。评测脚本 `outputs/viw_bootstrap_validation_20261123/evaluate_batch.py`，旧 D455 标定 `eval/depth_offset.json`；该标定不可直接用于新30绝对精度。
- 更详细逐项数字和因果纠正：`outputs/secone_init_diagnosis/validation/authorized_trials_20261124.md`。实验源码构建日志：同一 `validation/` 下的 `build_*.log`。最新隔离二进制在已停止容器的 `/root/catkin_ws/devel/lib/vins/viwo_node`，挂载源码就是上述 `source/`；现有副本包含众多环境变量控制的假设实验，不是可交付实现。

### 本轮核心结果（所有数字均来自隔离试验）

1. **不能靠阈值或多算几次解决。** 新30 scene20 无试验开关为 0 VIO 帧；只改运动预测前端，scene20/31/4 均 0。只改短轨迹 SfM，scene4 有 43 帧，scene20/31 仍 0。ORB 取点、ORB+短轨迹、光流丢点后的描述子找回及其组合都没救回 scene20/31；描述子后补找回仅 58/8206、72/8966 次。放开 SfM 两端平移方向，scene20 仍 0，scene31 仅末尾 2 帧。SfM 迭代 50→200、时间 0.2→1 秒，scene20 BA 代价中位数 0.603117→0.602958，仍 0 帧。
2. **能更早建图的组合不等于精度合格。** `VIW_SFM_CONVERGENCE_TEST=1 VIW_FREE_BASELINE_DIRECTION=1 VIW_INCREMENTAL_SFM=1 VIW_ROBUST_SFM=1` 在原始30 scene20/31 分别产生 25/58 VIO 帧，首次输出轮速约 13/25 km/h，目录 `ablate_robust_free_graph30/`。旧20 scene11/17 同组合产生 254/115 帧，但 scene17 最终仅 44 个地图点，D455 每帧仅 1–3 个有效比较点，不能用其 45–60% 百分比作稳定精度结论。旧20原版 scene17 有 284 个地图点、每帧 147–181 个 D455 有效点，误差约 3.0–3.2%。
3. **主要是长期地图/状态先验问题，不是尾端前端没有点。** 旧20 scene17 最后原版：总特征392、至少两帧352、正深度374、当前跟踪297、其中有效深度283；早初始化候选对应395、368、64、300、17（`trace_window_baseline20/`、`trace_window_anchor20/`）。两版最终滑窗时间跨度都约1.33秒，最后11次均选择边缘化次新帧，故不能简单说只因新版本抛弃了移动关键帧。直接加正深度/多帧重投影/视差检查不改善；无效点生命周期加坐标锚定后仅恢复到 scene17 当前有效深度17、评测6–7点，仍不合格。
4. **低速位姿受旧先验明显牵引。** 旧 scene17 最后0.667秒轮速积分约0.0067m；原版位姿差约0.003m，早初始化候选约0.15m。候选停车后每个轮速因子位置差约0.017–0.019m、加权残差末尾约51–56。轮速因子平移/旋转雅可比均经逐帧中心差分验证，最大误差约2.7e-6/8.0e-6（`trace_wheel_rot_jac20/`）。后端求解预算由8次/0.04秒增至20次/0.2秒没有改善残差或地图；视觉 Huber 换 Cauchy 也没有改善。输入 bag 轮速方向正确。
5. **仅为定位而跳过旧边缘化先验，效果最强，但不能作为正式修复。** `VIW_SKIP_PRIOR_BACKEND=1` 仍用当前滑窗视觉、IMU、轮速因子：scene17 轮速加权残差中位33.07→11.14、末尾51.09→6.39，当前有效深度17→267，输出地图点20→275；D455 每帧148/163/184点，误差5.74/5.80/5.98%，仍高于原版2.999/3.043/3.202%。scene11 同法误差约0.845/1.062/0.960%（105/132/122点），优于原版2.644/2.520/2.438%（115/126/132点）。新30 scene20/31 同法仍25/58 VIO帧，地图点260/237、当前有效深度245/209（`ablate_skip_prior30/`）。在跳过先验时，再移除坐标锚定/精确子空间开关，scene17结果完全相同（`ablate_skip_prior_free20/`），确认旧先验是主要冲突来源。
6. **旧先验中的视觉部分有明显责任，但不是全部。** 只不把旧视觉观测压缩进先验，当前视觉因子仍参与：scene17 末尾轮速加权残差51.09→20.393、当前有效深度17→210、输出点20→218；中位轮速残差33.07→27.57。最后一次实验目录 `ablate_skip_visual_margin20/scene_17/log/`，已跑完，未做D455深度评测（用户已要求停止）。剩余先验中的IMU/轮速/平面或整体局部参数化仍可能不一致，不能直接删除视觉先验交付。
7. `RefineGravityWithWheel` 在重力切平面每轮改变后仍累计旧法方程，数学上不对；隔离版每轮清零（`VIW_RESET_GRAVITY_NORMAL`）对 scene17 深度仅有有限改善，不能解释主因。`VIW_QUATERNION_PRIOR` 修正了四元数先验数值差分误差，但 scene17精度也几乎不变。`VIW_SKIP_NEAR_REFINE` 影响很小。详数见试验报告。

### 目录索引（逐模式原始日志都在 `validation/模式名/scene_N/log/`）

- 30 代表场景：`ablate_baseline30`(20:0帧)，`ablate_frontend30`(4/20/31:0)，`ablate_graph30`(4:43、20/31:0)，`ablate_orb30`、`ablate_orb_graph30`(4/20/31:0)，`ablate_reassoc30`、`ablate_reassoc_graph30`、`ablate_orb_reassoc_graph30`(20/31:0)，`ablate_free30`、`ablate_free_graph30`(20:0、31:2)，`ablate_convergence30`(20/31:0)、`ablate_convergence_graph30`(20:0、31:2)，`ablate_robust_free_graph30`(20:25、31:58)，`ablate_skip_prior30`(20:25、31:58)。这些不是全44批量验收。
- 旧20回归/消融：`ablate_baseline20`(11/17原版对照)，`ablate_frontend20`、`ablate_robust_free_graph20`、`ablate_gravity20`、`ablate_landmark20`、`ablate_landmark_state20`、`ablate_no_near20`、`ablate_anchor_landmark20`、`ablate_anchor_no_near20`、`ablate_prior_landmark20`、`ablate_solver_02_20`、`ablate_cauchy_visual20`、`ablate_skip_prior20`、`ablate_skip_prior20_scene11`、`ablate_skip_prior_free20`、`ablate_skip_visual_margin20`。旧20结果 JSON 在 `outputs/viw_bootstrap_validation_20261123/eval_模式名/results.json`；`ablate_cauchy_visual20` 和 `ablate_skip_visual_margin20` 仅有日志和地图点数，没有单独D455评测。`ablate_graph20`、`ablate_orb20`、`ablate_reassoc20`、`ablate_gravity30`、`ablate_solver_05_20` 只是准备了模式目录，**没有场景运行**。
- 只读诊断：`trace_map_baseline20`、`trace_map_anchor20`、`trace_window_baseline20`、`trace_window_anchor20`、`trace_wheel_baseline20`、`trace_wheel_anchor20`、`trace_wheel_jac20`、`trace_wheel_rot_jac20`。日志中的 `[MAP_LIFECYCLE]`、`[WHEEL_LINK]`、`[WHEEL_JAC]` 可直接核数。

### 接手应做什么

先检查`MarginalizationInfo`如何把旧视觉/IMU/轮速/平面因子压成先验，以及 7维位姿按6维切空间处理与实验版最老位姿2维锚定的相容性、世界坐标重置、先验线性化点和边缘化时各因子是否重复/漏算。用实际 scene17 先验做有限差分和逐因子残差/信息矩阵审计，找出数学错误并**正确保留先验**。不得把 `VIW_SKIP_PRIOR_BACKEND` 或 `VIW_SKIP_VISUAL_MARGIN` 作为最终方案；它们只是定位证据。之后必须原始30全44段、旧20全20段/60帧逐场景复测初始化时间、地图点覆盖率、D455误差；没有新30 D455跨相机外参和时钟标定前，不宣称新30绝对深度精度合格。50 km/h 尚未跑。用户当前要求先停，**不要自动恢复实验**。
