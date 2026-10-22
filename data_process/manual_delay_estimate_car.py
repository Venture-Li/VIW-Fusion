#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动指定单个轮速文件和单个IMU文件，计算二者之间的时延。
支持车上IMU格式（制表符分隔，无表头，9列：ts, col1, col2, ax, ay, az, gx, gy, gz）。
用法：修改下方 PATH 配置后直接运行。
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy import signal
from scipy.signal import butter, filtfilt
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 手动指定文件路径 ====================
WHEEL_FILE = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/time/wheel/combine.txt"   # 轮速文件（tab分隔）
IMU_FILE   = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/time/imu_20260821_152841_710.txt"  # 车上IMU文件（制表符分隔，9列）
LABEL      = "manual_pair_vehicle"
# =========================================================

# 算法参数
RESAMPLE_HZ = 100
MAX_LAG_SEC = 2.0
PLOT_ENABLE = True

# 多频段分析: 尝试多个频段，找最可靠的
# 每个元素: (low_hz, high_hz, label)
MULTI_BAND_FREQS = [
    (0.1,  1.0,  "0.1-1.0Hz"),
    (0.15, 2.0,  "0.15-2.0Hz"),
    (0.5,  3.0,  "0.5-3.0Hz"),
    (0.5,  5.0,  "0.5-5.0Hz"),
    (1.0,  5.0,  "1.0-5.0Hz"),
    (1.0,  8.0,  "1.0-8.0Hz"),
]
# 相关性阈值: 某频段 cc r > 此值 且 cc与gcc一致 才认为可靠
CC_RELIABLE_THRESHOLD = 0.35
# cc 和 gcc 在同一频段结果相差超过此值，认为 cc 不可靠
CC_GCC_MAX_DIFF_MS = 100

# ---------- 函数定义 ----------

def load_wheel(filepath):
    """加载轮速: ts_ns, vx, vy, vz, wz (tab分隔)"""
    data = np.loadtxt(filepath)
    ts = data[:, 0].astype(np.int64)
    vx = data[:, 1]
    return ts, vx

def load_imu(filepath):
    """
    加载车上IMU: 制表符分隔，无表头，9列:
    timestamp, col1, col2, ax, ay, az, gx, gy, gz
    返回时间戳(纳秒)和 ax (前向加速度)
    """
    data = np.loadtxt(filepath, delimiter='\t')  # 制表符分隔
    ts = data[:, 0].astype(np.int64)            # 第0列时间戳
    ax = data[:, 3]                             # 第4列（索引3）是 ax
    return ts, ax

def resample(t_grid, ts_src, values):
    return np.interp(t_grid, ts_src.astype(np.float64), values)

def refine_peak(lags_roi, corr_roi):
    idx = np.argmax(np.abs(corr_roi))
    if idx == 0 or idx == len(corr_roi) - 1:
        return lags_roi[idx]
    x = lags_roi[idx-1:idx+2]
    y = corr_roi[idx-1:idx+2]
    A = np.vstack([x**2, x, np.ones(3)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b, _ = coeffs
    if abs(a) < 1e-12:
        return lags_roi[idx]
    x_peak = -b / (2 * a)
    if x_peak < lags_roi[idx-1] or x_peak > lags_roi[idx+1]:
        return lags_roi[idx]
    return x_peak

def cross_correlation(sig_a, sig_b, dt, max_lag_sec):
    sig_a = sig_a - np.mean(sig_a)
    sig_b = sig_b - np.mean(sig_b)
    corr = np.correlate(sig_a, sig_b, mode='full')
    n = len(sig_a)
    lags = np.arange(-n + 1, n) * dt
    max_lag_samples = int(max_lag_sec / dt)
    center = n - 1
    lo = max(0, center - max_lag_samples)
    hi = min(len(corr), center + max_lag_samples + 1)
    corr_roi = corr[lo:hi]
    lags_roi = lags[lo:hi]
    best_lag = refine_peak(lags_roi, corr_roi)
    std_a = np.std(sig_a)
    std_b = np.std(sig_b)
    if std_a * std_b == 0:
        corr_coef = 0.0
    else:
        corr_coef = np.max(np.abs(corr_roi)) / (std_a * std_b * n)
    return best_lag, corr_coef, lags_roi, corr_roi

def gcc_phat(sig_a, sig_b, dt, max_lag_sec):
    sig_a = sig_a - np.mean(sig_a)
    sig_b = sig_b - np.mean(sig_b)
    n = len(sig_a)
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    X = np.fft.rfft(sig_a, n=n_fft)
    Y = np.fft.rfft(sig_b, n=n_fft)
    G = X * np.conj(Y)
    eps = 1e-10
    G_phat = G / (np.abs(G) + eps)
    corr = np.fft.irfft(G_phat, n=n_fft)[:2 * n - 1]
    lags = np.arange(-n + 1, n) * dt
    max_lag_samples = int(max_lag_sec / dt)
    center = n - 1
    lo = max(0, center - max_lag_samples)
    hi = min(len(corr), center + max_lag_samples + 1)
    corr_roi = corr[lo:hi]
    lags_roi = lags[lo:hi]
    best_lag = refine_peak(lags_roi, corr_roi)
    peak_val = np.max(np.abs(corr_roi))
    return best_lag, peak_val, lags_roi, corr_roi

def make_bandpass_filter(low, high, fs, order=4):
    """创建巴特沃斯带通滤波器"""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    def apply(sig):
        return filtfilt(b, a, sig)
    return apply

def multi_band_delay(wheel_vx_rs, imu_ax_rs, dt, max_lag_sec, freq_bands):
    """
    多频段时延分析: 在多个频段上分别用互相关和GCC-PHAT估计时延，
    返回所有结果，以便找到最可靠的一致估计。
    """
    # 方法2: 速度 (积分IMU ax)
    imu_vx = cumulative_trapezoid(imu_ax_rs, dx=dt, initial=0.0)
    imu_vx = signal.detrend(imu_vx)
    wheel_vx_dt = signal.detrend(wheel_vx_rs)

    all_results = []

    for low, high, label in freq_bands:
        bp_filter = make_bandpass_filter(low, high, int(1.0 / dt))

        # ---- 方法2: 速度 ----
        wheel_vx_bp = bp_filter(wheel_vx_dt)
        imu_vx_bp = bp_filter(imu_vx)

        cc2_lag, cc2_corr, _, _ = cross_correlation(wheel_vx_bp, imu_vx_bp, dt, max_lag_sec)
        gcc2_lag, gcc2_peak, _, _ = gcc_phat(wheel_vx_bp, imu_vx_bp, dt, max_lag_sec)

        # ---- 方法1: 加速度 ----
        wheel_acc_bp = np.gradient(wheel_vx_bp, dt)
        imu_ax_bp = bp_filter(imu_ax_rs)  # 直接滤波IMU原始ax

        cc1_lag, cc1_corr, _, _ = cross_correlation(wheel_acc_bp, imu_ax_bp, dt, max_lag_sec)
        gcc1_lag, gcc1_peak, _, _ = gcc_phat(wheel_acc_bp, imu_ax_bp, dt, max_lag_sec)

        all_results.append({
            'band': label,
            'cc_vel_lag': cc2_lag, 'cc_vel_r': cc2_corr,
            'gcc_vel_lag': gcc2_lag, 'gcc_vel_peak': gcc2_peak,
            'cc_acc_lag': cc1_lag, 'cc_acc_r': cc1_corr,
            'gcc_acc_lag': gcc1_lag, 'gcc_acc_peak': gcc1_peak,
        })

    return all_results

def find_best_delay(all_results, cc_threshold):
    """
    从多频段结果中找最可靠的时延估计。
    策略:
      1. 优先找互相关 r > cc_threshold 的频段
      2. 如果多个频段满足，取它们GCC-PHAT一致的那个
      3. 如果都不满足，找所有GCC-PHAT最一致的频段
    """
    print(f"\n  {'─'*70}")
    print(f"  {'频段':>12s}  {'cc_acc':>10s} {'gcc_acc':>10s}  {'cc_vel':>10s} {'gcc_vel':>10s}")
    print(f"  {'─'*70}")

    for r in all_results:
        print(f"  {r['band']:>12s}  {r['cc_acc_lag']*1000:+8.1f}ms {r['gcc_acc_lag']*1000:+8.1f}ms  "
              f"{r['cc_vel_lag']*1000:+8.1f}ms {r['gcc_vel_lag']*1000:+8.1f}ms  "
              f"(r_acc={r['cc_acc_r']:.3f}, r_vel={r['cc_vel_r']:.3f})")

    # 筛选互相关可靠的频段:
    # 1) r > cc_threshold
    # 2) 同频段 cc 和 gcc 结果相差 < CC_GCC_MAX_DIFF_MS (排除低频漂移虚假相关)
    cc_reliable = []
    for r in all_results:
        if r['cc_acc_r'] > cc_threshold and abs(r['cc_acc_lag'] - r['gcc_acc_lag']) * 1000 < CC_GCC_MAX_DIFF_MS:
            cc_reliable.append(('acc', r))
        if r['cc_vel_r'] > cc_threshold and abs(r['cc_vel_lag'] - r['gcc_vel_lag']) * 1000 < CC_GCC_MAX_DIFF_MS:
            cc_reliable.append(('vel', r))

    if cc_reliable:
        # 有可靠互相关: 取r最高的那个
        best = max(cc_reliable, key=lambda x: x[1][f'cc_{x[0]}_r'])
        mode, r = best
        if mode == 'acc':
            best_lag = r['cc_acc_lag']
            best_method = f"cc_acc@{r['band']}"
            best_score = r['cc_acc_r']
        else:
            best_lag = r['cc_vel_lag']
            best_method = f"cc_vel@{r['band']}"
            best_score = r['cc_vel_r']
        print(f"\n  互相关可靠 → 最佳: {best_method} lag={best_lag*1000:+.1f}ms (r={best_score:.3f})")
        return best_lag, best_score, best_method

    # 互相关都不可靠, 找GCC-PHAT最一致的频段
    # 收集所有GCC-PHAT结果
    gcc_lags = []
    for r in all_results:
        gcc_lags.append(r['gcc_acc_lag'])
        gcc_lags.append(r['gcc_vel_lag'])

    gcc_lags = np.array(gcc_lags)
    gcc_median = np.median(gcc_lags)
    gcc_std = np.std(gcc_lags)
    gcc_range = np.max(gcc_lags) - np.min(gcc_lags)

    # 排除离群值后取平均
    inliers = gcc_lags[np.abs(gcc_lags - gcc_median) < 2 * gcc_std]
    if len(inliers) > 0:
        gcc_mean = np.mean(inliers)
    else:
        gcc_mean = gcc_median

    print(f"\n  ⚠ 所有频段互相关 r < {cc_threshold}, 改用 GCC-PHAT")
    print(f"  GCC-PHAT 所有估计: {np.array2string(gcc_lags*1000, precision=1, suppress_small=True)} ms")
    print(f"  GCC-PHAT 中位数: {gcc_median*1000:+.1f}ms, 标准差: {gcc_std*1000:.1f}ms, 范围: {gcc_range*1000:.1f}ms")

    if gcc_range < 0.05:  # 所有GCC结果在50ms内一致
        print(f"  GCC-PHAT 高度一致 → 时延 = {gcc_mean*1000:+.1f}ms")
        return gcc_mean, gcc_std, 'gcc_consensus'
    elif gcc_range < 0.2:
        print(f"  GCC-PHAT 基本一致 → 时延 = {gcc_mean*1000:+.1f}ms (范围={gcc_range*1000:.0f}ms)")
        return gcc_mean, gcc_std, 'gcc_weak_consensus'
    else:
        print(f"  ⚠ GCC-PHAT 不一致 (范围={gcc_range*1000:.0f}ms), 时延估计不可靠!")
        print(f"  建议: 选取有更多加减速变化的行驶段重新标定")
        return gcc_mean, gcc_std, 'gcc_unreliable'

def estimate_delay(wheel_file, imu_file, label=""):
    print(f"\n{'='*60}")
    print(f"分析: {label}")
    print(f"  轮速: {os.path.basename(wheel_file)}")
    print(f"  IMU:  {os.path.basename(imu_file)}")
    print(f"{'='*60}")

    wheel_ts, wheel_vx = load_wheel(wheel_file)
    imu_ts, imu_ax = load_imu(imu_file)

    # 只取有运动的数据段
    moving_mask = np.abs(wheel_vx) > 0.1
    if not np.any(moving_mask):
        print("  ⚠ 轮速全为零，跳过")
        return None

    moving_indices = np.where(moving_mask)[0]
    start_idx = max(0, moving_indices[0] - 100)
    end_idx = min(len(wheel_vx) - 1, moving_indices[-1] + 100)
    wheel_ts = wheel_ts[start_idx:end_idx + 1]
    wheel_vx = wheel_vx[start_idx:end_idx + 1]

    print(f"  轮速: {len(wheel_ts)} 条, "
          f"{wheel_ts[0]/1e9:.3f}s ~ {wheel_ts[-1]/1e9:.3f}s")
    print(f"  IMU:  {len(imu_ts)} 条, "
          f"{imu_ts[0]/1e9:.3f}s ~ {imu_ts[-1]/1e9:.3f}s")

    # 统一重采样
    dt = 1.0 / RESAMPLE_HZ
    dt_ns = int(1e9 / RESAMPLE_HZ)
    t_start = max(wheel_ts[0], imu_ts[0])
    t_end = min(wheel_ts[-1], imu_ts[-1])
    t_grid = np.arange(t_start, t_end, dt_ns, dtype=np.int64)

    if len(t_grid) < 100:
        print("  ⚠ 重叠时间太短，跳过")
        return None

    wheel_vx_rs = resample(t_grid, wheel_ts, wheel_vx)
    imu_ax_rs = resample(t_grid, imu_ts, imu_ax)

    duration = (t_end - t_start) / 1e9
    print(f"  重叠时长: {duration:.1f}s, 样本数: {len(t_grid)}")

    # ---- 多频段时延分析 ----
    print(f"\n  [多频段分析] 轮速 vs IMU 时延估计")
    all_results = multi_band_delay(wheel_vx_rs, imu_ax_rs, dt, MAX_LAG_SEC, MULTI_BAND_FREQS)

    best_lag, best_score, best_method = find_best_delay(all_results, CC_RELIABLE_THRESHOLD)

    if best_lag > 0:
        print(f"\n  解释: 轮速领先 IMU {best_lag*1000:.1f}ms")
    else:
        print(f"\n  解释: IMU 领先轮速 {-best_lag*1000:.1f}ms")

    # ---- 绘图 ----
    if PLOT_ENABLE:
        t_sec = (t_grid - t_grid[0]).astype(np.float64) / 1e9

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delay_plots")
        os.makedirs(out_dir, exist_ok=True)

        # 画原始信号
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ax1 = axes[0]
        ax1.plot(t_sec, -imu_ax_rs, label='-IMU ax (raw)', alpha=0.7, linewidth=0.5)
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title('Raw IMU acceleration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = axes[1]
        ax2.plot(t_sec, wheel_vx_rs, label='Wheel vx (raw)', alpha=0.7, linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Raw wheel speed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{label}_raw.png"), dpi=150)
        plt.close()

        # 画多频段滤波后信号对比 (取最佳频段)
        best_band = best_method.split('@')[-1] if '@' in best_method else None
        if best_band:
            for r in all_results:
                if r['band'] == best_band:
                    break
            else:
                r = all_results[0]
        else:
            r = all_results[0]

        bp_filter = make_bandpass_filter(
            float(r['band'].split('-')[0]),
            float(r['band'].split('-')[1].replace('Hz','')),
            int(1.0/dt))
        imu_vx = cumulative_trapezoid(imu_ax_rs, dx=dt, initial=0.0)
        wheel_vx_f = bp_filter(signal.detrend(wheel_vx_rs))
        imu_vx_f = bp_filter(signal.detrend(imu_vx))
        wheel_acc_f = np.gradient(wheel_vx_f, dt)
        imu_ax_f = bp_filter(imu_ax_rs)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ax1 = axes[0]
        ax1.plot(t_sec, wheel_vx_f, label='Wheel vx (filtered)', alpha=0.8)
        ax1.plot(t_sec, imu_vx_f, label='IMU vx (filtered)', alpha=0.8)
        ax1.set_title(f'Filtered velocity ({r["band"]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = axes[1]
        ax2.plot(t_sec, wheel_acc_f, label='Wheel acc (filtered)', alpha=0.8)
        ax2.plot(t_sec, -imu_ax_f, label='-IMU ax (filtered)', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_title(f'Filtered acceleration ({r["band"]})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{label}_filtered.png"), dpi=150)
        plt.close()

        print(f"  图表已保存: {os.path.join(out_dir, f'{label}_raw.png')}")
        print(f"  图表已保存: {os.path.join(out_dir, f'{label}_filtered.png')}")

    return {
        'label': label,
        'best_lag_ms': best_lag * 1000,
        'best_method': best_method,
        'best_score': best_score,
    }

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("手动指定模式：单对文件时延估计（车上IMU格式）")
    result = estimate_delay(WHEEL_FILE, IMU_FILE, LABEL)
    if result:
        print(f"\n最终结果：滞后 = {result['best_lag_ms']:.1f} ms")
    else:
        print("计算失败，请检查数据或路径。")