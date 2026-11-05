# SVC 深度评测

批量入口调用 `eval_depth_svc.py`，不再嵌入重复 Python 代码。依赖 Python 3、numpy、opencv-python。

## 必须确认的标定

创建本地 JSON（不要填写未经确认的数值）：

```json
{
  "K_depth": [[386.7249, 0, 323.3481], [0, 386.7249, 233.6760], [0, 0, 1]],
  "R_depth_to_processed": null,
  "t_depth_to_processed_m": null,
  "depth_scale_m": 0.001,
  "depth_time_offset_s": null,
  "svc_time_offset_s": 0.0
}
```

内参和深度单位也必须与本次 depth0 数据一致。输入深度必须是该内参对应的无畸变 Z 深度（不是距离、不是另一相机坐标系的 aligned depth）。

变换定义：`P_processed = R_depth_to_processed @ P_depth + t_depth_to_processed_m`。
processed 是 **SVC 光心、处理脚本旋转后的轴向**，不是原始 SVC，也不是 D455 光心。

若现有标定 `P_D = R_sd P_S + t_sd` 的 D **就是 depth0 光学系**，且处理脚本使用相同 R_sd，则 R_depth_to_processed = I，t_depth_to_processed_m = -t_sd。

若 D 是 RGB 光学系，还需设备标定 `P_depth = R_dc P_color + t_dc`。此时：

```
R_depth_to_processed = R_dc.T
t_depth_to_processed_m = -R_dc.T @ t_dc - t_sd
```

这里不能把 RGB→depth 变换省略。参数必须对应相同安装和采集设备。

时间偏移定义：`corrected_file_timestamp = file_timestamp + offset`，目标为 pose_output/图像时间戳时钟。不能直接照搬 SVC-IMU 的 td。若已在导出数据中同步，则填 0；否则填写测得偏移。

## 运行

```bash
DATA_ROOT_BASE=/path/to/data bash eval/batch_eval_svc.sh --calibration /path/to/calibration.json
```

可用环境变量 `OUTPUT_DIR`、`CONFIG` 覆盖输出目录与 **本次运行使用的** SLAM 配置。默认评测 pose_output 第三行；`--frame-index 0` 改为第一行。默认匹配时间差上限 10ms；可用 `--max-time-diff 0.005` 收紧。

若相机外参在线优化，必须提供最终外参 YAML：`--camera-extrinsics /path/to/final.yaml`，包含 body_T_cam0；不要使用初始外参代替优化后的值。每个场景最终外参不同时，请逐场景调用 Python 入口。

## 输出与限制

- registered_depth_svc.npy：深度反投影后变换至 processed SVC，再用 z-buffer 投影生成的 Z 深度（米），空洞为 0。
- color_depth_overlay.png：使用上述配准深度叠加，不是直接混合原始深度图。
- depth_comparison.csv：同一 processed 像素下估计与实测 Z 深度；剔除空洞和局部深度突变。
- depth_eval_result.json：统计、选帧信息、标定和时间差。
- depth_eval.log 与 summary_svc.csv：运行日志和成功/失败状态，失败返回非零。

不插值填洞，不用 SLAM 的估计深度选择真值位置，避免估计误差影响对应点。仅处理深度相机自身可见表面的 z-buffer；两相机遮挡差异、标定残差、像素取整仍会引入误差。程序没有运动补偿；即使时间差小于阈值，也不代表曝光时刻完全相同。高速运动应收紧阈值或另行提供轨迹进行运动补偿。
