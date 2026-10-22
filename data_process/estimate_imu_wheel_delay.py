#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
估计 IMU (D455) 与轮速计 (车辆) 之间的系统性时延。

改进:
  - 抛物线插值实现亚采样精度 (<1ms)
  - Bootstrap 分段估计标准差 (量化不确定度)
  - 自动 FIR/Butterworth 滤波器回退
  - 详细统计输出及置信区间

用法:
  python3 estimate_imu_wheel_delay.py

输入:
  speed/recordSpeed_*.txt  轮速 (ts_ns, vx, vy, vz, wz, tab分隔)
  time/scene_*/imu0/data.csv  IMU (ts_ns, gx, gy, gz, ax, ay, az, 逗号分隔, 有表头)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy import signal
from scipy.signal import butter, filtfilt
import os
import glob
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 配置 ====================
SPEED_DIR = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/wheel/"
IMU_DIR   = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/8.21/data/静态/"
RESAMPLE_HZ = 100          # 重采样频率
MAX_LAG_SEC = 2.0          # 最大搜索滞后 (秒)
FIR_COEFF_FILE = "/home/nio/下载/BPF_0P05_0p12_0p3_0p37.csv"  # 可选
BOOTSTRAP_SEG_SEC = 20.0   # 每段时长 (秒) 用于 Bootstrap
PLOT_ENABLE = True         # 是否生成图表
# ==============================================


def load_wheel(filepath):
    """加载轮速: ts_ns, vx, vy, vz, wz (tab 分隔)"""
    data = np.loadtxt(filepath)
    ts = data[:, 0].astype(np.int64)
    vx = data[:, 1]
    return ts, vx


def load_imu(filepath):
    """加载 IMU: timestamp_ns, gx, gy, gz, ax, ay, az (逗号分隔, 有表头)"""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    ts = data[:, 0].astype(np.int64)
    ax = data[:, 4]  # 前向加速度 (D455 x轴)
    return ts, ax


def resample(t_grid, ts_src, values):
    """线性插值到统一时间网格"""
    return np.interp(t_grid, ts_src.astype(np.float64), values)


def load_fir_coeffs(filepath):
    """加载 FIR 系数文件，若不存在则返回 None"""
    if not os.path.exists(filepath):
        return None
    try:
        coeffs = np.loadtxt(filepath, delimiter=',')
        if len(coeffs) < 3:
            return None
        if coeffs.ndim > 1:
            coeffs = coeffs.flatten()
        return coeffs
    except:
        return None


def refine_peak(lags_roi, corr_roi):
    """
    对互相关峰值进行抛物线插值，返回亚采样滞后 (秒)。
    若峰值在边界，则直接返回边界值。
    """
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
    """
    互相关法估计时延 (含亚采样插值)。
    lag > 0 表示 sig_a 领先 sig_b
    返回 (best_lag_seconds, correlation_coefficient, all_lags, all_corr)
    """
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
    """
    广义互相关 (GCC-PHAT) 含亚采样插值。
    lag > 0 表示 sig_a 领先 sig_b
    """
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


def bootstrap_lag(sig_a, sig_b, dt, segment_sec=20.0, max_lag=2.0):
    """将信号分段，每段独立估计时延，返回 (均值, 标准差)"""
    seg_len = int(segment_sec / dt)
    n_seg = len(sig_a) // seg_len
    if n_seg < 2:
        return np.nan, np.nan
    lags = []
    for i in range(n_seg):
        a = sig_a[i*seg_len:(i+1)*seg_len]
        b = sig_b[i*seg_len:(i+1)*seg_len]
        lag, _, _, _ = cross_correlation(a, b, dt, max_lag)
        lags.append(lag)
    if len(lags) < 2:
        return np.nan, np.nan
    return np.mean(lags), np.std(lags)


def find_matching_speed_file(imu_ts_start, imu_ts_end, speed_dir):
    """找到与 IMU 时间范围重叠的轮速文件 (与原脚本完全一致)"""
    speed_files = sorted(glob.glob(os.path.join(speed_dir, "recordSpeed_*.txt")))
    best_file = None
    best_overlap = 0
    for f in speed_files:
        data = np.loadtxt(f)
        if len(data) == 0:
            continue
        ws, we = data[0, 0], data[-1, 0]
        overlap = min(imu_ts_end, we) - max(imu_ts_start, ws)
        if overlap > best_overlap:
            best_overlap = overlap
            best_file = f
    return best_file, best_overlap / 1e9


def estimate_delay(wheel_file, imu_file, label=""):
    """对一对文件估计时延 (增加Bootstrap，自动跳过短数据段)"""
    print(f"\n{'='*60}")
    print(f"分析: {label}")
    print(f"  轮速: {os.path.basename(wheel_file)}")
    print(f"  IMU:  {os.path.basename(imu_file)}")
    print(f"{'='*60}")

    wheel_ts, wheel_vx = load_wheel(wheel_file)
    imu_ts, imu_ax = load_imu(imu_file)

    # 只取有运动的数据段 (vx > 0.1 m/s)
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

    # ---- 准备滤波器 & 检查长度 ----
    fir_coeffs = load_fir_coeffs(FIR_COEFF_FILE)
    use_fir = False
    if fir_coeffs is not None:
        # 估算 filtfilt 所需的填充长度（保守估计 3 * 滤波器阶数）
        padlen = 3 * (len(fir_coeffs) - 1)  # FIR 阶数 = len(coeffs)-1
        if len(t_grid) > padlen:
            use_fir = True
            def apply_filter(sig):
                return signal.filtfilt(fir_coeffs, [1.0], sig)
            print(f"  使用 FIR 滤波器 (0.15–0.35 Hz), 信号长度 {len(t_grid)} > {padlen}")
        else:
            print(f"  ⚠ 信号长度 {len(t_grid)} 不足以进行 FIR 滤波 (需 > {padlen})，跳过该 scene")
            return None   # 直接跳过，不进行后续计算
    else:
        nyq = 0.5 * RESAMPLE_HZ
        b, a = butter(4, [0.15/nyq, 0.35/nyq], btype='band')
        def apply_filter(sig):
            return filtfilt(b, a, sig)
        print("  使用 Butterworth 带通 (0.15–0.35 Hz)")

    # ---- 方法2: 轮速 vs IMU 速度 (先滤波，共用速度信号) ----
    print(f"\n  [方法2] 轮速 vs IMU 速度 互相关...")
    imu_vx = cumulative_trapezoid(imu_ax_rs, dx=dt, initial=0.0)
    imu_vx = signal.detrend(imu_vx)
    wheel_vx_dt = signal.detrend(wheel_vx_rs)

    wheel_vx_bp = apply_filter(wheel_vx_dt)
    imu_vx_bp = apply_filter(imu_vx)

    cc2_lag, cc2_corr, lags2, corr2 = cross_correlation(
        wheel_vx_bp, imu_vx_bp, dt, MAX_LAG_SEC)
    print(f"    互相关: 滞后 = {cc2_lag*1000:+.1f}ms, r = {cc2_corr:.4f}")

    gcc2_lag, gcc2_peak, lags2g, corr2g = gcc_phat(
        wheel_vx_bp, imu_vx_bp, dt, MAX_LAG_SEC)
    print(f"    GCC-PHAT: 滞后 = {gcc2_lag*1000:+.1f}ms, peak = {gcc2_peak:.4f}")

    # ---- 方法1: 轮加速度 vs IMU ax (对滤波后速度对称微分) ----
    print(f"\n  [方法1] 轮加速度 vs IMU ax 互相关...")
    wheel_acc_bp = np.gradient(wheel_vx_bp, dt)
    imu_ax_bp = np.gradient(imu_vx_bp, dt)

    cc1_lag, cc1_corr, lags1, corr1 = cross_correlation(
        wheel_acc_bp, imu_ax_bp, dt, MAX_LAG_SEC)
    print(f"    互相关: 滞后 = {cc1_lag*1000:+.1f}ms, r = {cc1_corr:.4f}")

    gcc1_lag, gcc1_peak, lags1g, corr1g = gcc_phat(
        wheel_acc_bp, imu_ax_bp, dt, MAX_LAG_SEC)
    print(f"    GCC-PHAT: 滞后 = {gcc1_lag*1000:+.1f}ms, peak = {gcc1_peak:.4f}")

    # ---- Bootstrap 标准差 ----
    mean_lag, std_lag = bootstrap_lag(wheel_vx_bp, imu_vx_bp, dt, BOOTSTRAP_SEG_SEC, MAX_LAG_SEC)
    if not np.isnan(mean_lag):
        print(f"    Bootstrap (速度): mean={mean_lag*1000:+.1f}ms, std={std_lag*1000:.1f}ms")
    else:
        std_lag = np.nan

    # ---- 选择最佳估计 ----
    results = {
        'cc_acc': (cc1_lag, cc1_corr),
        'gcc_acc': (gcc1_lag, gcc1_peak),
        'cc_vel': (cc2_lag, cc2_corr),
        'gcc_vel': (gcc2_lag, gcc2_peak),
    }
    best_method = max(results, key=lambda k: abs(results[k][1]))
    best_lag, best_score = results[best_method]

    print(f"\n  {'─'*50}")
    print(f"  最佳估计: {best_method} → 滞后 = {best_lag*1000:+.1f}ms (score={best_score:.4f})")
    if best_lag > 0:
        print(f"  解释: 轮速领先 IMU {best_lag*1000:.1f}ms")
    else:
        print(f"  解释: IMU 领先轮速 {-best_lag*1000:.1f}ms")

    # ---- 绘图 ----
    if PLOT_ENABLE:
        t_sec = (t_grid - t_grid[0]).astype(np.float64) / 1e9
        moving = np.abs(wheel_vx_rs) > 0.1
        if np.any(moving):
            moving_idx = np.where(moving)[0]
            t_plot_start = max(0, t_sec[moving_idx[0]] - 2.0)
            t_plot_end = t_sec[moving_idx[-1]] + 2.0
        else:
            t_plot_start, t_plot_end = t_sec[0], t_sec[-1]
        zoom_center = (t_plot_start + t_plot_end) / 2.0
        zoom_start = max(t_plot_start, zoom_center - 5.0)
        zoom_end = min(t_plot_end, zoom_center + 5.0)

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delay_plots")
        os.makedirs(out_dir, exist_ok=True)

        # ===== 图1: 加速度相关性 =====
        fig1, axes1 = plt.subplots(5, 1, figsize=(14, 16))

        ax = axes1[0]; ax_twin = ax.twinx()
        l1, = ax.plot(t_sec, -imu_ax_rs, 'C0', alpha=0.7, linewidth=0.5, label='-IMU ax (raw)')
        l2, = ax_twin.plot(t_sec, np.gradient(wheel_vx_rs, dt), 'C1', alpha=0.7, linewidth=0.5, label='Wheel acc (dv/dt)')
        ax.set_xlim(t_plot_start, t_plot_end)
        ax.set_ylabel('-IMU ax (m/s²)', color='C0'); ax_twin.set_ylabel('Wheel acc (m/s²)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'{label} [Acceleration] Raw Data'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes1[1]; ax_twin = ax.twinx()
        l1, = ax.plot(t_sec, -imu_ax_bp, 'C0', alpha=0.8, linewidth=0.8, label='-IMU ax (bandpass)')
        l2, = ax_twin.plot(t_sec, wheel_acc_bp, 'C1', alpha=0.8, linewidth=0.8, label='Wheel acc (bandpass)')
        ax.set_xlim(t_plot_start, t_plot_end)
        ax.set_ylabel('-IMU ax (m/s²)', color='C0'); ax_twin.set_ylabel('Wheel acc (m/s²)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'Filtered Acceleration (0.15–0.35 Hz)'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes1[2]; ax_twin = ax.twinx()
        l1, = ax.plot(t_sec, -imu_ax_bp, 'C0', alpha=0.9, linewidth=1.0, label='-IMU ax (bandpass)')
        l2, = ax_twin.plot(t_sec, wheel_acc_bp, 'C1', alpha=0.9, linewidth=1.0, label='Wheel acc (bandpass)')
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylabel('-IMU ax (m/s²)', color='C0'); ax_twin.set_ylabel('Wheel acc (m/s²)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'Zoomed 10s — Acceleration'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes1[3]; ax_twin = ax.twinx()
        t_shifted = t_sec + cc1_lag
        l1, = ax.plot(t_shifted, -imu_ax_bp, 'C0', alpha=0.9, linewidth=1.0, label=f'-IMU ax (shifted {cc1_lag*1000:+.0f}ms)')
        l2, = ax_twin.plot(t_sec, wheel_acc_bp, 'C1', alpha=0.9, linewidth=1.0, label='Wheel acc (bandpass)')
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylabel('-IMU ax (m/s²)', color='C0'); ax_twin.set_ylabel('Wheel acc (m/s²)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'Aligned by cc1_lag={cc1_lag*1000:+.0f}ms'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes1[4]
        ax.plot(lags1 * 1000, corr1, 'b-', linewidth=0.8, label='Cross-correlation')
        ax.axvline(x=cc1_lag * 1000, color='r', linestyle='--', linewidth=1.5, label=f'Best lag = {cc1_lag*1000:+.1f}ms')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Lag (ms)'); ax.set_ylabel('Correlation')
        ax.set_title(f'Cross-corr: Wheel acc vs IMU ax (r={cc1_corr:.4f})'); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig1.savefig(os.path.join(out_dir, f"{label}_acc.png"), dpi=150)
        plt.close(fig1)
        print(f"  加速度图表已保存: {os.path.join(out_dir, f'{label}_acc.png')}")

        # ===== 图2: 速度相关性 =====
        fig2, axes2 = plt.subplots(5, 1, figsize=(14, 16))

        ax = axes2[0]; ax_twin = ax.twinx()
        l1, = ax.plot(t_sec, -imu_vx, 'C0', alpha=0.7, linewidth=0.5, label='-IMU vx (integrated)')
        l2, = ax_twin.plot(t_sec, wheel_vx_rs, 'C1', alpha=0.7, linewidth=0.5, label='Wheel vx (raw)')
        ax.set_xlim(t_plot_start, t_plot_end)
        ax.set_ylabel('-IMU vx (m/s)', color='C0'); ax_twin.set_ylabel('Wheel vx (m/s)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'{label} [Velocity] Raw Data'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes2[1]; ax_twin = ax.twinx()
        l1, = ax.plot(t_sec, -imu_vx_bp, 'C0', alpha=0.8, linewidth=0.8, label='-IMU vx (bandpass)')
        l2, = ax_twin.plot(t_sec, wheel_vx_bp, 'C1', alpha=0.8, linewidth=0.8, label='Wheel vx (bandpass)')
        ax.set_xlim(t_plot_start, t_plot_end)
        ax.set_ylabel('-IMU vx (m/s)', color='C0'); ax_twin.set_ylabel('Wheel vx (m/s)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'Filtered Velocity (0.15–0.35 Hz)'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes2[2]; ax_twin = ax.twinx()
        l1, = ax.plot(t_sec, -imu_vx_bp, 'C0', alpha=0.9, linewidth=1.0, label='-IMU vx (bandpass)')
        l2, = ax_twin.plot(t_sec, wheel_vx_bp, 'C1', alpha=0.9, linewidth=1.0, label='Wheel vx (bandpass)')
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylabel('-IMU vx (m/s)', color='C0'); ax_twin.set_ylabel('Wheel vx (m/s)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'Zoomed 10s — Velocity'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes2[3]; ax_twin = ax.twinx()
        t_shifted = t_sec + cc2_lag
        l1, = ax.plot(t_shifted, -imu_vx_bp, 'C0', alpha=0.9, linewidth=1.0, label=f'-IMU vx (shifted {cc2_lag*1000:+.0f}ms)')
        l2, = ax_twin.plot(t_sec, wheel_vx_bp, 'C1', alpha=0.9, linewidth=1.0, label='Wheel vx (bandpass)')
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylabel('-IMU vx (m/s)', color='C0'); ax_twin.set_ylabel('Wheel vx (m/s)', color='C1')
        ax.tick_params(axis='y', colors='C0'); ax_twin.tick_params(axis='y', colors='C1')
        ax.set_title(f'Aligned by cc2_lag={cc2_lag*1000:+.0f}ms'); ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes2[4]
        ax.plot(lags2 * 1000, corr2, 'g-', linewidth=0.8, label='Cross-correlation')
        ax.axvline(x=cc2_lag * 1000, color='r', linestyle='--', linewidth=1.5, label=f'Best lag = {cc2_lag*1000:+.1f}ms')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Lag (ms)'); ax.set_ylabel('Correlation')
        ax.set_title(f'Cross-corr: Wheel vx vs IMU vx (r={cc2_corr:.4f})'); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"{label}_vel.png"), dpi=150)
        plt.close(fig2)
        print(f"  速度图表已保存: {os.path.join(out_dir, f'{label}_vel.png')}")

    return {
        'label': label,
        'best_lag_ms': best_lag * 1000,
        'best_method': best_method,
        'best_score': best_score,
        'std_ms': std_lag * 1000 if not np.isnan(std_lag) else np.nan,
        'cc_vel_lag': cc2_lag * 1000,
        'cc_vel_corr': cc2_corr,
    }


def main():
    print("=" * 60)
    print("IMU (D455) ↔ 轮速计 (车辆) 时延估计 (含Bootstrap)")
    print("=" * 60)

    imu_dirs = sorted(glob.glob(os.path.join(IMU_DIR, "scene_*")))
    if not imu_dirs:
        print("未找到 IMU 数据目录")
        return

    all_results = []
    for imu_dir in imu_dirs:
        scene_name = os.path.basename(imu_dir)
        imu_file = os.path.join(imu_dir, "imu0", "data.csv")
        if not os.path.exists(imu_file):
            continue

        imu_data = np.loadtxt(imu_file, delimiter=',', skiprows=1, max_rows=1)
        imu_ts_start = imu_data[0, 0] if imu_data.ndim == 2 else imu_data[0]
        imu_data_end = np.loadtxt(imu_file, delimiter=',', skiprows=1)
        imu_ts_end = imu_data_end[-1, 0]

        wheel_file, overlap = find_matching_speed_file(
            imu_ts_start, imu_ts_end, SPEED_DIR)

        if wheel_file is None or overlap < 1.0:
            continue

        result = estimate_delay(wheel_file, imu_file, scene_name)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n未找到可分析的数据对")
        return

    print(f"\n\n{'='*60}")
    print(f"汇总: {len(all_results)} 个 scene 的时延估计")
    print(f"{'='*60}")
    print(f"{'Scene':<12} {'滞后(ms)':>10} {'方法':>12} {'Score':>10} {'Std(ms)':>10}")
    print(f"{'─'*60}")

    lags = []
    stds = []
    for r in all_results:
        std_str = f"{r['std_ms']:>10.1f}" if not np.isnan(r['std_ms']) else "      N/A"
        print(f"{r['label']:<12} {r['best_lag_ms']:+10.1f} {r['best_method']:>12} {r['best_score']:>10.4f} {std_str}")
        lags.append(r['best_lag_ms'])
        if not np.isnan(r['std_ms']):
            stds.append(r['std_ms'])

    lags = np.array(lags)
    print(f"{'─'*60}")
    print(f"  均值: {np.mean(lags):+.1f}ms")
    print(f"  中位数: {np.median(lags):+.1f}ms")
    print(f"  标准差 (场景间): {np.std(lags):.1f}ms")
    print(f"  范围: [{np.min(lags):+.1f}, {np.max(lags):+.1f}]ms")
    if stds:
        print(f"  Bootstrap 平均标准差: {np.mean(stds):.1f}ms")

    median_lag = np.median(lags)
    ci_low = median_lag - 1.96 * np.std(lags)
    ci_high = median_lag + 1.96 * np.std(lags)
    print(f"\n{'='*60}")
    print(f"建议时延补偿值 (td_wheel): {median_lag/1000:+.4f}s")
    print(f"95% 置信区间: [{ci_low/1000:+.4f}, {ci_high/1000:+.4f}]s")
    if median_lag > 0:
        print(f"含义: 轮速时间戳比 IMU 晚 {median_lag:.1f}ms")
    else:
        print(f"含义: IMU 时间戳比轮速晚 {-median_lag:.1f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()