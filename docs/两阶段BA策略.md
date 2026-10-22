# VIW-Fusion 两阶段 BA 策略总结

## 概述

VIW-Fusion 在滑动窗口非线性优化中采用**两阶段 Bundle Adjustment** 策略，在每帧图像处理中依次执行：

1. **第一阶段（全量联合优化）**：联合优化所有状态变量，使用 Huber 鲁棒核函数
2. **第二阶段（近处点深度精修）**：固定位姿等状态，仅精修近处特征点的逆深度，使用 Cauchy 鲁棒核函数

---

## 整体流程

```
processImage()
  │
  ├─ 1. f_manager.triangulate()         —— 三角化新特征点
  ├─ 2. optimization()                  —— ★ 第一阶段：全量 BA
  ├─ 3. updateNearFlags()               —— 标记近处点（深度 < 3m）
  ├─ 4. refineNearPoints()              —— ★ 第二阶段：近处点深度精修
  ├─ 5. outliersRejection()             —— 外点剔除（平均重投影误差 > 5 像素）
  ├─ 6. f_manager.removeOutlier()       —— 移除被标记的外点
  ├─ 7. slideWindow()                   —— 滑动窗口边缘化
  └─ 8. f_manager.removeFailures()      —— 移除深度异常的特征
```

关键代码位于 `vins_estimator/src/estimator/estimator.cpp:940-989`：

```cpp
f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
optimization();  // 第一阶段全量优化

if (solver_flag == NON_LINEAR)
{
    updateNearFlags();   // 标记近处点
    refineNearPoints();  // 第二阶段深度精修
}
```

---

## 第一阶段：全量联合优化 (`optimization()`)

**代码位置**：`vins_estimator/src/estimator/estimator.cpp:1772-2054`

### 优化变量

| 变量 | 维度 | 说明 |
|------|------|------|
| `para_Pose[i]` | 7 (x,y,z,qx,qy,qz,qw) | 滑动窗口内所有帧的位姿 |
| `para_SpeedBias[i]` | 9 (vx,vy,vz,bax,bay,baz,bgx,bgy,bgz) | 速度与 IMU 偏置 |
| `para_Feature[i]` | 1 (inv_depth) | 所有特征点的逆深度 |
| `para_Ex_Pose[i]` | 7 | 相机-IMU 外参（条件性优化） |
| `para_Ex_Pose_wheel[0]` | 7 | 轮速计-IMU 外参（条件性优化） |
| `para_Ix_sx/sy/sw_wheel[0]` | 1×3 | 轮速计内参：尺度因子（条件性优化） |
| `para_plane_R[0]` | 3 | 平面法向量（条件性优化） |
| `para_plane_Z[0]` | 1 | 平面高度（条件性优化） |
| `para_Td[0]` | 1 | 相机时间偏移（条件性优化） |
| `para_Td_wheel[0]` | 1 | 轮速计时间偏移（条件性优化） |

### 残差项

| 残差因子 | 连接的参数块 | 说明 |
|----------|-------------|------|
| `IMUFactor` | Pose_i, SpeedBias_i, Pose_j, SpeedBias_j | IMU 预积分约束 |
| `WheelFactor` | Pose_i, Pose_j, Ex_Pose_wheel, Ix_sx, Ix_sy, Ix_sw, Td_wheel | 轮速预积分约束 |
| `PlaneFactor` | Pose_i, Ex_Pose_wheel, plane_R, plane_Z | 地平面约束 |
| `ProjectionTwoFrameOneCamFactor` | Pose_i, Pose_j, Ex_Pose, Feature, Td | 视觉重投影（单目跨帧） |
| `ProjectionTwoFrameTwoCamFactor` | Pose_i, Pose_j, Ex_Pose0, Ex_Pose1, Feature, Td | 视觉重投影（双目跨帧） |
| `ProjectionOneFrameTwoCamFactor` | Ex_Pose0, Ex_Pose1, Feature, Td | 视觉重投影（双目同帧） |
| `MarginalizationFactor` | 上次边缘化保留的参数块 | 边缘化先验 |

### 求解器配置

| 参数 | 值 |
|------|-----|
| 鲁棒核函数 | `HuberLoss(1.0)` |
| 线性求解器 | `DENSE_SCHUR` |
| 信任域策略 | `DOGLEG` |
| 最大迭代次数 | 8~10（可配置 `max_num_iterations`） |
| 最大求解时间 | 0.06s（可配置 `max_solver_time`） |

---

## 近处点标记：`updateNearFlags()`

**代码位置**：`vins_estimator/src/estimator/estimator.cpp:992-1019`

在两次 BA 之间执行，将当前帧深度 < 3m 的特征点标记为 `is_near = true`。

```cpp
// 将特征点从世界坐标系转换到当前帧相机坐标系
Vector3d pts_cur = R_cj_w * w_pts + t_cj_w;
double depth_cur = pts_cur.z();

it_per_id.is_near = (depth_cur > 0 && depth_cur < NEAR_DEPTH_THRESHOLD);  // 默认 3.0m
```

---

## 第二阶段：近处点深度精修 (`refineNearPoints()`)

**代码位置**：`vins_estimator/src/estimator/estimator.cpp:1022-1172`

### 核心思想

第一阶段完成后，位姿估计已经较为准确。第二阶段将所有位姿/速度/外参/内参**固定**，释放全部计算资源专注于精修近处点的深度——这些近处点对定位精度至关重要。

### 固定所有非深度参数

```cpp
// 固定位姿、速度偏置
for (int i = 0; i <= frame_count; i++) {
    problem.SetParameterBlockConstant(para_Pose[i]);
    problem.SetParameterBlockConstant(para_SpeedBias[i]);
}
// 固定外参、内参、平面参数、时间偏移
problem.SetParameterBlockConstant(para_Ex_Pose[i]);
problem.SetParameterBlockConstant(para_Ex_Pose_wheel[0]);
problem.SetParameterBlockConstant(para_Ix_sx_wheel[0]);
problem.SetParameterBlockConstant(para_Ix_sy_wheel[0]);
problem.SetParameterBlockConstant(para_Ix_sw_wheel[0]);
problem.SetParameterBlockConstant(para_plane_R[0]);
problem.SetParameterBlockConstant(para_plane_Z[0]);
problem.SetParameterBlockConstant(para_Td[0]);
problem.SetParameterBlockConstant(para_Td_wheel[0]);
```

### 速度自适应点选择

根据当前车速动态调整精修点的比例，兼顾精度与计算效率：

```
选择比例 p = 线性插值(NEAR_POINT_RATIO_MAX → NEAR_POINT_RATIO_MIN)

停车时 (speed ≤ 0.05 m/s):  p = 0.8  (选 80% 的近处点)
高速时 (speed ≥ 5.0 m/s):   p = 0.2  (选 20% 的近处点)
中速时:                     p = 线性插值
```

```cpp
double cur_speed = Vs[frame_count].norm();
if (cur_speed >= MAX_SPEED_FOR_RATIO)       p = NEAR_POINT_RATIO_MIN;   // 0.2
else if (cur_speed <= STOP_SPEED)             p = NEAR_POINT_RATIO_MAX;   // 0.8
else                                          p = 插值
```

选择时按当前帧深度**从小到大排序**，优先选取最近的近处点。

### 仅添加视觉重投影残差

第二阶段只构建视觉重投影因子（`ProjectionTwoFrameOneCamFactor`），不添加 IMU、轮速、平面等因子。

### 求解器配置

| 参数 | 值 |
|------|-----|
| 鲁棒核函数 | `CauchyLoss(1.0)` |
| 线性求解器 | `DENSE_SCHUR` |
| 最大迭代次数 | 5 |
| 输出到控制台 | 关闭 |

### 深度写回

优化后，将逆深度转换回深度并写回 `FeaturePerId::estimated_depth`，限制在 `[0.1, 50.0]` 范围内。

---

## 两阶段对比

| 特性 | 第一阶段（全量 BA） | 第二阶段（深度精修） |
|------|-------------------|---------------------|
| **优化变量** | 所有状态（位姿、速度、偏置、外参、内参、深度） | 仅近处点的逆深度 |
| **鲁棒核函数** | `HuberLoss(1.0)` | `CauchyLoss(1.0)` |
| **最大迭代次数** | 8~10 | 5 |
| **选点策略** | 所有观测 ≥2 次的特征点 | 仅近处点（深度 < 3m），按速度自适应比例选取 |
| **残差因子** | IMU + 轮速 + 平面 + 视觉 + 边缘化先验 | 仅视觉重投影 |
| **速度自适应** | 无 | 停车 80% / 高速 20% |

### 鲁棒核函数选择

- **HuberLoss**：残差较小时为二次增长（L2），较大时线性增长（L1）。适合第一阶段，因为此时位姿和深度都未精确，需要在抑制离群点的同时保留一定的梯度信息。
- **CauchyLoss**：比 Huber 更激进地抑制大残差。适合第二阶段，因为位姿已经固定，大残差更可能是真正的离群点。

---

## 后处理：外点剔除 (`outliersRejection()`)

**代码位置**：`vins_estimator/src/estimator/estimator.cpp:2519-2577`

在两次 BA 完成后执行：

- 对每个特征点，计算其所有观测的平均重投影误差
- 阈值：`ave_err * FOCAL_LENGTH > 5` 像素
- 超过阈值的特征点被标记为外点并移除

---

## 深度门控 (`setDepth()`)

**代码位置**：`vins_estimator/src/estimator/feature_manager.cpp:142-196`

BA 优化后对深度进行双重检查：

| 检查类型 | 范围 | 说明 |
|----------|------|------|
| start-frame 深度 | [0.3, 20.0] | 宽松阈值，远处特征提供旋转约束 |
| current-frame 深度 | [0.3, 10.0] | 严格阈值，避免近处误匹配 |

如果当前帧深度计算异常（点在相机后方），退回到 start-frame 深度检查兜底。

---

## 可配置参数

**配置文件**：`config/realsense_d455i/realsense_stereo_imu_config_d455.yaml:101-105`

```yaml
near_depth_threshold: 3.0     # 近处点深度阈值（米）
near_point_ratio_min: 0.2     # 高速时近处点选择比例下限
near_point_ratio_max: 0.8     # 停车时近处点选择比例上限
max_speed_for_ratio: 5.0      # 比例计算对应的最大速度（m/s，≈18 km/h）
```

**代码读取**：`vins_estimator/src/estimator/estimator.cpp:278-282`

```cpp
ros::param::param<double>("near_depth_threshold", NEAR_DEPTH_THRESHOLD, 3.0);
ros::param::param<double>("near_point_ratio_min", NEAR_POINT_RATIO_MIN, 0.2);
ros::param::param<double>("near_point_ratio_max", NEAR_POINT_RATIO_MAX, 0.8);
ros::param::param<double>("max_speed_for_ratio", MAX_SPEED_FOR_RATIO, 5.0);
```

---

## 设计动机

1. **为什么需要第二阶段？** 第一阶段全量 BA 中，位姿、速度、偏置、外参、内参、深度等所有变量耦合在一起，优化自由度极高。在第一阶段收敛后固定位姿，对近处点做第二轮精修，可以更充分地利用视觉信息，获得更精确的深度估计。

2. **为什么只优化近处点？** 近处特征点的深度精度对定位影响最大——深度误差对重投影误差的贡献与距离成反比。远处点的深度误差对位姿影响较小，不值得投入额外计算。

3. **为什么速度自适应？** 高速行驶时，近处点快速移出视野，深度精修的边际收益降低；停车时（如泊车场景），近处点稳定且对精度要求更高，应投入更多计算资源。

4. **为什么第二阶段用 CauchyLoss？** 第一阶段 HuberLoss 在抑制离群点的同时保留一定梯度，适合位姿未收敛时。第二阶段位姿已固定，大残差极可能是真正的离群点，CauchyLoss 的重尾特性可以更激进地排除它们。