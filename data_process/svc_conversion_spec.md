# SVC鱼眼图像 → D455针孔图像转换

## 1. 任务要求

| 项目 | 说明 |
|------|------|
| 输入 | SVC Right 相机鱼眼图像，JPG格式，1920×1536 |
| 输出 | 针孔投影图像，PNG格式，640×480 |
| 目标 | 与 D455 RGB 相机相同的分辨率和近似 FOV，用于 Depth-Anything-3 深度估计和 VIW-Fusion SLAM |
| 输入路径 | `scene_N/svc/CAMERA0_FRAME_<ms_timestamp>.jpg` |
| 输出路径 | `scene_N/svc_processed/<ms_timestamp>000000.png` |

## 2. 参数来源

**弃用了 `ex_svc_right.json` 中的 12 参数多项式畸变模型**，改用棋盘格标定板重新标定得到的 KB4 参数。

标定脚本：
- `extrinsic_calib_svc_d455_0821.py` — 0821 数据标定
- `extrinsic_calib_svc_d455_set2.py` — 第 2 组数据标定（含 `ExtrinsicCalibrator` 类）

标定结果文件：
- `extrinsic_svc_d455_0821.txt` — 最终标定结果

## 3. 相机内参

### 3.1 SVC KB4 参数 (棋盘格标定获得)

来自 `extrinsic_svc_d455_0821.txt`，使用 `cv2.fisheye` (Kannala-Brandt, 4 参数) 模型：

```
K_SVC:
  fx = 413.4521
  fy = 412.6878
  cx = 959.5152
  cy = 768.0989

dist_SVC (Kb 4): [0.17030521, -0.00308843, -0.01711661, 0.00219803]

SVC Kb 内参标定 RMS = 0.2071 px
```

> 畸变模型: `r = fx * (θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹)`

### 3.2 D455 基准内参 (640×480 针孔)

来自 `intrinsics.txt` (RealSense 格式)：

```
K_D455:
  fx = 381.6182
  fy = 381.2833
  cx = 320.0508
  cy = 241.8335

dist_D455 (Plumb-Bob 5): [-0.05834, 0.06961, 0.00011, -0.00060, -0.02226]
```

### 3.3 D455→SVC 外参

来自 `extrinsic_svc_d455_0821.txt`，`P_d455 = R @ P_svc + t`：

```
R = [[ 0.9988213  -0.01314425 -0.04672521]
     [ 0.01744383  0.99552857  0.09283629]
     [ 0.04529602 -0.09354193  0.99458443]]

t = [-0.337857, -0.048525, 0.163150] m

欧拉角 (xyz/RPY): [-5.37°, -2.60°, 1.00°]
```

> 此外参已用于 VIW-Fusion 配置文件 `realsense_stereo_imu_config_d455_svc.yaml` 中的 `body_T_cam0`。

## 4. 目标内参计算

不直接使用 D455 基准内参，而是基于 D455 基准内参的 FOV 按比例扩大视野：

| 参数 | 值 | 说明 |
|------|-----|------|
| TOP_RATIO | 1.25 | 顶部视野扩大 1.25 倍 |
| SIDE_RATIO | 1.2 | 左右视野扩大 1.2 倍 |
| 底部 | 1.0 | 保持不变 |

**计算过程**:

```
H = 479, W = 639   (OUT_H-1, OUT_W-1)

# 从 D455 基准 K 计算各方向 tan(θ)
tan_top   = cy0 / fy0
tan_bot   = (H - cy0) / fy0
tan_left  = cx0 / fx0
tan_right = (W - cx0) / fx0

# 按比例扩大
tan_top_new   = tan_top * TOP_RATIO
tan_bot_new   = tan_bot
tan_left_new  = tan_left * SIDE_RATIO
tan_right_new = tan_right * SIDE_RATIO

# 计算新 K
fy_new = H / (tan_top_new + tan_bot_new)
fx_new = W / (tan_left_new + tan_right_new)
cy_new = fy_new * tan_top_new
cx_new = cx0   # 保持水平主点不变
```

## 5. 转换流程

```
加载 SVC KB4 内参 (从 extrinsic_svc_d455_0821.txt)
        │
        ▼
计算目标针孔内参 (基于 D455 基准 + TOP_RATIO/SIDE_RATIO 扩大)
        │
        ▼
cv2.fisheye.initUndistortRectifyMap(
    K_SVC, DIST_SVC, R=eye(3), P=K_target,
    size=(640, 480), dtype=CV_32FC1)
        │
        ▼
遍历 scene_N/svc/ 目录下所有 JPG:
  cv2.remap(img, map1, map2, INTER_LANCZOS4) → 输出 PNG
  如果图像尺寸 ≠ 1920x1536，先底部补黑边
```

## 6. 处理参数

| 参数 | 值 |
|------|-----|
| 输出分辨率 | 640 × 480 |
| 插值方法 | `cv2.INTER_LANCZOS4` |
| 边界模式 | `BORDER_CONSTANT`, 填充黑色 |
| 输出格式 | PNG |
| 输出命名 | `<timestamp_ms>000000.png` |
| 并行 | 支持 multiprocessing (`--workers N`) |

## 7. 脚本使用

```bash
# 单个 scene
conda run -n da python3 process_svc_to_d455.py \
  --scene-dir /path/to/scene_1

# 批量 scene (多进程)
conda run -n da python3 process_svc_to_d455.py \
  --data-dir /path/to/parent \
  --workers 4

# 选项
--output-suffix svc_processed   # 输出子目录名 (默认)
--svc-size 1920x1536           # SVC 原始图像尺寸 (默认)
```

## 8. 相关文件

| 文件 | 用途 |
|------|------|
| `process_svc_to_d455.py` | 图像转换脚本 |
| `extrinsic_calib_svc_d455_0821.py` | 0821 数据外参标定脚本 |
| `extrinsic_calib_svc_d455_set2.py` | 第 2 组数据外参标定脚本 (含 `ExtrinsicCalibrator` 类) |
| `extrinsic_svc_d455_0821.txt` | 标定结果 (内参 + 外参) |
| `extrinsic_svc_d455_0821.npz` | 标定结果 (NumPy 格式) |
| `ex_svc_right.json` | SVC 出厂标定参数 (已弃用) |
| `D455.txt` | D455 设备信息 (参考) |
| `../config/realsense_d455i/realsense_stereo_imu_config_d455_svc.yaml` | VIW-Fusion 配置文件 (使用标定外参) |