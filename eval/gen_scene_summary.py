import os
import re
import csv
import numpy as np
from openpyxl import Workbook

# 输出Excel文件路径
out_xlsx = "/home/nio/vscode/python_project_backup/VIW-Fusion/output/scene_summary.xlsx"

# 需要统计的scene目录
base_dir = "/home/nio/vscode/python_project_backup/VIW-Fusion/output"
scene_dirs = [d for d in os.listdir(base_dir) if d.startswith("scene_") and os.path.isdir(os.path.join(base_dir, d))]
scene_dirs.sort(key=lambda x: int(x.split("_")[-1]))

# Excel表头
header = [
    "scene", "初始化尝试次数", "初始化成功",
    "绝对误差均值", "绝对误差中位数", "绝对误差最大值",
    "相对误差均值", "相对误差中位数", "相对误差最大值",
    "散点图斜率", "总帧数"
]

def parse_init_log(log_dir):
    # 解析初始化日志，返回初始化尝试次数和是否成功
    files = [f for f in os.listdir(log_dir) if f.startswith("vins_init_log_")]
    if not files:
        return ["", ""]
    files.sort()
    with open(os.path.join(log_dir, files[0]), "r") as f:
        txt = f.read()
    blocks = re.split(r'=+ initialStructure\(\) START =+=', txt)
    last = blocks[-1] if len(blocks) > 1 else ''
    succ = re.search(r'success: (\d+)', last)
    idx = len(blocks) - 1
    return [
        str(idx),
        "是" if succ and succ.group(1) == "1" else "否"
    ]

def parse_depth_csv(csv_path):
    # 统计评测误差和斜率
    abs_errs, rel_errs, est_depths, gt_depths = [], [], [], []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                abs_errs.append(float(row["abs_error(m)"]))
                rel_errs.append(float(row["rel_error"]))
                est_depths.append(float(row["est_depth(m)"]))
                gt_depths.append(float(row["gt_depth(m)"]))
        abs_errs = np.array(abs_errs)
        rel_errs = np.array(rel_errs)
        est_depths = np.array(est_depths)
        gt_depths = np.array(gt_depths)
        slope = np.linalg.lstsq(gt_depths.reshape(-1, 1), est_depths, rcond=None)[0][0]
        return [
            np.mean(abs_errs), np.median(abs_errs), np.max(abs_errs),
            np.mean(rel_errs), np.median(rel_errs), np.max(rel_errs),
            slope
        ]
    except Exception:
        return [""] * 7

def count_frames(data_csv):
    # 统计帧数
    try:
        with open(data_csv, "r") as f:
            return sum(1 for _ in f) - 1  # 去掉表头
    except Exception:
        return ""

wb = Workbook()
ws = wb.active
ws.append(header)

for scene in scene_dirs:
    row = [scene]
    # 1. 初始化参数
    log_dir = os.path.join(base_dir, scene, "log")
    row += parse_init_log(log_dir) if os.path.exists(log_dir) else ["", ""]
    # 2. 评测误差
    depth_csv = os.path.join(base_dir, scene, "depth_comparison.csv")
    row += parse_depth_csv(depth_csv) if os.path.exists(depth_csv) else [""] * 7
    # 3. 帧数
    scene_num = scene.split("_")[-1]
    data_csv = f"/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/data/scene_{scene_num}/cam0/data.csv"
    row.append(count_frames(data_csv))
    ws.append(row)

wb.save(out_xlsx)
print("统计完成，结果已保存到", out_xlsx)
