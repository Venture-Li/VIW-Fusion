#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVC(针孔+Kb 鱼眼) -> D455(针孔+Plumb-Bob) 外参标定 - 第 2 组数据
求解 T = [R|t]:  P_d455 = R @ P_svc + t

本文件独立运行, 不修改 extrinsic_calib_svc_d455.py (第 1 组)。
标定逻辑封装为 ExtrinsicCalibrator 类, 结果输出到:
  - extrinsic_svc_d455_set2.txt
  - extrinsic_svc_d455_set2.npz
"""
import cv2
import numpy as np
import glob
import os
import json
import re
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class ExtrinsicCalibrator:
    """SVC(Kb 鱼眼) -> D455(Plumb-Bob 针孔) 外参标定器

    参数:
        d455_intrinsics_path: D455 内参文件 (RealSense 格式)
        svc_intrinsics_path:  SVC 内参文件 (JSON, 含出厂主点)
        d455_img_dir:          D455 标定图像目录
        svc_img_dir:           SVC 标定图像目录 (crop 图)
        checkerboard:          棋盘格内角点数 (列, 行)
        square_size:           棋盘格每格边长 (米)
        out_npz / out_txt:     输出文件路径
    """

    NS_TO_MS = 1_000_000
    PAIR_MAX_MS_DIFF = 100
    SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    SVC_KEEP_RMS  = 1.5     # SVC PnP RMS 剔除阈值 (px)
    D455_KEEP_RMS = 0.8     # D455 PnP RMS 剔除阈值 (px)
    REJECT_RATIO  = 1.5     # 第二阶段: 残差 > RATIO * 中位数 视为离群
    MIN_KEEP      = 8       # 第二阶段至少保留的对数

    def __init__(self, d455_intrinsics_path, svc_intrinsics_path,
                 d455_img_dir, svc_img_dir,
                 checkerboard=(9, 6), square_size=0.037,
                 out_npz="extrinsic_svc_d455_set2.npz",
                 out_txt="extrinsic_svc_d455_set2.txt"):
        self.d455_intrinsics = d455_intrinsics_path
        self.svc_intrinsics  = svc_intrinsics_path
        self.d455_img_dir    = d455_img_dir
        self.svc_img_dir     = svc_img_dir
        self.checkerboard    = checkerboard
        self.square_size     = square_size
        self.out_npz         = out_npz
        self.out_txt         = out_txt

        # 标定结果 (run 后填充)
        self.K_svc = self.dist_svc = None
        self.K_d455 = self.dist_d455 = None
        self.rms_svc_intrinsic = None
        self.R_opt = self.t_opt = self.rvec_opt = None
        self.rms_opt = None
        self.n_pairs = None
        self.pair_err_lines = []
        self.svc_mean = self.svc_rms = None
        self.d455_mean = self.d455_rms = None

    # ---------- 棋盘格 3D 点 ----------
    def make_objp(self):
        cb = self.checkerboard
        objp = np.zeros((cb[0] * cb[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:cb[0], 0:cb[1]].T.reshape(-1, 2) * self.square_size
        return objp

    # ---------- D455 内参解析 (针孔 + Plumb-Bob) ----------
    def parse_d455_intrinsics(self, path):
        with open(path, "r") as f:
            text = f.read()
        fx, fy = map(float, re.search(r"fx, fy\s*:\s*([\d.\-eE]+),\s*([\d.\-eE]+)", text).groups())
        cx, cy = map(float, re.search(r"cx, cy\s*:\s*([\d.\-eE]+),\s*([\d.\-eE]+)", text).groups())
        coeffs_str = re.search(r"Coeffs\s*:\s*\[([^\]]+)\]", text).group(1)
        coeffs = np.array([float(x) for x in coeffs_str.split(",")], dtype=np.float64)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = coeffs.reshape(-1, 1)
        return K, dist

    # ---------- SVC 出厂主点解析 ----------
    def parse_svc_principal(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        cx, cy = data["intrinsic_param"]["principal_point"]
        W = data["intrinsic_param"]["camera_width"]
        H = data["intrinsic_param"]["camera_height"]
        return float(cx), float(cy), (int(W), int(H))

    # ---------- 角点检测 ----------
    def detect_corners(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
        if not ret:
            return gray.shape[::-1], None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.SUBPIX_CRIT)
        return gray.shape[::-1], corners.reshape(-1, 2)

    # ---------- SVC Kb 内参标定 (scipy 联合优化) ----------
    def calibrate_svc_kb(self, img_dir, cx_factory, cy_factory, img_size):
        objp = self.make_objp()
        objpoints, imgpoints = [], []
        files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        for fname in files:
            _, corners = self.detect_corners(fname)
            if corners is None:
                print(f"  [SVC 内参] 角点检测失败: {os.path.basename(fname)}")
                continue
            objpoints.append(objp)
            imgpoints.append(corners)
        if not objpoints:
            raise RuntimeError("SVC crop 图未检测到任何棋盘角点")
        n = len(objpoints)
        print(f"  [SVC 内参] 成功 {n} / {len(files)} 张")

        ret_pin, K_pin, _, rvecs_pin, tvecs_pin = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None)
        print(f"  [SVC 内参] 针孔初值 RMS={ret_pin:.4f}  fx={K_pin[0,0]:.2f} fy={K_pin[1,1]:.2f} "
              f"cx={K_pin[0,2]:.2f} cy={K_pin[1,2]:.2f}")
        print(f"  [SVC 内参] 出厂主点: cx={cx_factory:.2f} cy={cy_factory:.2f} (仅参考)")

        obj_f = [o.reshape(-1, 1, 3).astype(np.float64) for o in objpoints]
        img_f = [c.reshape(-1, 1, 2).astype(np.float64) for c in imgpoints]

        def residuals(x):
            fx, fy, cx, cy = x[0], x[1], x[2], x[3]
            D = x[4:8].reshape(4, 1)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            off_r = 8; off_t = 8 + 3 * n
            res = []
            for i in range(n):
                rvec = x[off_r + 3 * i: off_r + 3 * i + 3]
                tvec = x[off_t + 3 * i: off_t + 3 * i + 3]
                proj, _ = cv2.fisheye.projectPoints(obj_f[i], rvec, tvec, K, D)
                res.append((proj.reshape(-1, 2) - img_f[i].reshape(-1, 2)).ravel())
            return np.concatenate(res)

        x0 = np.concatenate([
            [K_pin[0, 0], K_pin[1, 1], K_pin[0, 2], K_pin[1, 2]],
            np.zeros(4),
            np.array(rvecs_pin).ravel(),
            np.array(tvecs_pin).ravel(),
        ])
        res = least_squares(residuals, x0, method="lm", max_nfev=10000)

        fx, fy, cx, cy = res.x[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = res.x[4:8].reshape(4, 1)
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        print(f"  [SVC 内参] Kb 联合优化 RMS={rms:.4f}")
        print(f"  [SVC 内参] Kb 最终  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
        print(f"  [SVC 内参] Kb dist = {dist.ravel()}")
        return K, dist, rms

    # ---------- 按时间戳配对图像 ----------
    def pair_images(self, d455_dir, svc_dir):
        def ts_d455(name):
            n = Path(name).stem
            return int(n) // self.NS_TO_MS if n.isdigit() else None
        def ts_svc(name):
            n = Path(name).stem
            p = "CAMERA0_FRAME_"
            return int(n[len(p):]) if n.startswith(p) and n[len(p):].isdigit() else None
        d455 = sorted([(ts_d455(f), os.path.join(d455_dir, f))
                       for f in os.listdir(d455_dir) if ts_d455(f)])
        svc  = sorted([(ts_svc(f),  os.path.join(svc_dir,  f))
                       for f in os.listdir(svc_dir)  if ts_svc(f)])
        pairs, used_b = [], set()
        j = 0
        for ta, pa in d455:
            best_j, best_diff = -1, float('inf')
            for k in range(max(0, j - 1), len(svc)):
                diff = abs(ta - svc[k][0])
                if diff < best_diff:
                    best_diff, best_j = diff, k
                if svc[k][0] > ta and diff > best_diff:
                    break
            if best_j >= 0 and best_diff <= self.PAIR_MAX_MS_DIFF and best_j not in used_b:
                pairs.append((pa, svc[best_j][1], ta))
                used_b.add(best_j)
                j = best_j
        return pairs

    # ---------- 联合 BA 残差 (SVC + D455 双相机) ----------
    def residuals(self, x, n_pairs, objp, corners_a, corners_b,
                  K_a, dist_a, K_b, dist_b):
        """a = SVC (fisheye), b = D455 (pinhole Plumb-Bob)
        变量布局: rvec_rel(3), t_rel(3), rvec_a_i(3*n), tvec_a_i(3*n)
        """
        rvec_rel = x[:3]
        t_rel    = x[3:6]
        R_rel, _ = cv2.Rodrigues(rvec_rel)
        off_r = 6
        off_t = 6 + 3 * n_pairs
        res = []
        for i in range(n_pairs):
            rvec_i = x[off_r + 3*i : off_r + 3*i + 3]
            tvec_i = x[off_t + 3*i : off_t + 3*i + 3]
            proj_a, _ = cv2.fisheye.projectPoints(
                objp.reshape(-1, 1, 3).astype(np.float64),
                rvec_i, tvec_i, K_a, dist_a)
            res.append((proj_a.reshape(-1, 2) - corners_a[i]).ravel())
            R_i, _ = cv2.Rodrigues(rvec_i)
            R_b = R_rel @ R_i
            t_b = (R_rel @ tvec_i.reshape(3, 1) + t_rel.reshape(3, 1)).ravel()
            rvec_b, _ = cv2.Rodrigues(R_b)
            proj_b, _ = cv2.projectPoints(objp, rvec_b, t_b, K_b, dist_b)
            res.append((proj_b.reshape(-1, 2) - corners_b[i]).ravel())
        return np.concatenate(res)

    # ---------- 单次 BA (支持 Cauchy 鲁棒损失) ----------
    def _run_ba(self, n_in, c_a, c_b, rv_a, tv_a, R_init, t_init,
                K_svc, dist_svc, K_d455, dist_d455, objp, use_cauchy=True):
        rvec_init, _ = cv2.Rodrigues(R_init)
        x0 = np.concatenate([
            rvec_init.ravel(), t_init,
            np.array(rv_a).ravel(), np.array(tv_a).ravel(),
        ])
        if use_cauchy:
            kwargs = dict(method="trf", max_nfev=20000,
                          args=(n_in, objp, c_a, c_b, K_svc, dist_svc, K_d455, dist_d455),
                          loss="cauchy", f_scale=1.0)
        else:
            kwargs = dict(method="lm", max_nfev=20000,
                          args=(n_in, objp, c_a, c_b, K_svc, dist_svc, K_d455, dist_d455))
        r = least_squares(self.residuals, x0, **kwargs)
        rv = r.x[:3]; tv = r.x[3:6]
        R_, _ = cv2.Rodrigues(rv)
        off_r_ = 6; off_t_ = 6 + 3 * n_in
        per_pair = []
        for i in range(n_in):
            rvec_i = r.x[off_r_ + 3*i : off_r_ + 3*i + 3]
            tvec_i = r.x[off_t_ + 3*i : off_t_ + 3*i + 3]
            proj_a, _ = cv2.fisheye.projectPoints(
                objp.reshape(-1, 1, 3).astype(np.float64),
                rvec_i, tvec_i, K_svc, dist_svc)
            ea = (proj_a.reshape(-1, 2) - c_a[i]).ravel()
            R_i, _ = cv2.Rodrigues(rvec_i)
            R_b = R_ @ R_i
            t_b = (R_ @ tvec_i.reshape(3, 1) + tv.reshape(3, 1)).ravel()
            rvec_b, _ = cv2.Rodrigues(R_b)
            proj_b, _ = cv2.projectPoints(objp, rvec_b, t_b, K_d455, dist_d455)
            eb = (proj_b.reshape(-1, 2) - c_b[i]).ravel()
            per_pair.append(float(np.sqrt((np.sum(ea**2) + np.sum(eb**2))
                                          / (2 * len(ea)))))
        rms_total = float(np.sqrt(np.mean(r.fun ** 2)))
        return r, R_, tv, rms_total, per_pair

    # ---------- 主流程 ----------
    def run(self):
        objp = self.make_objp()

        # ---- 1. 标定 SVC 内参 ----
        print("=" * 60)
        print("1. 标定 SVC 内参 (针孔 + Kb)")
        print("=" * 60)
        cx, cy, svc_size = self.parse_svc_principal(self.svc_intrinsics)
        print(f"  出厂主点: cx={cx:.4f} cy={cy:.4f}  分辨率={svc_size}")
        self.K_svc, self.dist_svc, self.rms_svc_intrinsic = self.calibrate_svc_kb(
            self.svc_img_dir, cx, cy, svc_size)
        print(f"  K_svc =\n{self.K_svc}")
        print(f"  dist_svc = {self.dist_svc.ravel()}")

        # ---- 2. 读取 D455 内参 ----
        print("\n" + "=" * 60)
        print("2. 读取 D455 内参 (针孔 + Plumb-Bob)")
        print("=" * 60)
        self.K_d455, self.dist_d455 = self.parse_d455_intrinsics(self.d455_intrinsics)
        print(f"  K_d455 =\n{self.K_d455}")
        print(f"  dist_d455 = {self.dist_d455.ravel()}")

        # ---- 3. 配对 + 角点检测 + PnP ----
        print("\n" + "=" * 60)
        print("3. 配对图像并检测角点")
        print("=" * 60)
        pairs = self.pair_images(self.d455_img_dir, self.svc_img_dir)
        print(f"  配对数: {len(pairs)}")

        corners_a_list, corners_b_list = [], []   # a=SVC, b=D455
        rvecs_a, tvecs_a, rvecs_b, tvecs_b = [], [], [], []
        used_pairs = []
        for pa, pb, ts in pairs:   # pa=D455, pb=SVC
            _, ca = self.detect_corners(pb)   # SVC 角点
            _, cb = self.detect_corners(pa)   # D455 角点
            if ca is None or cb is None:
                print(f"  跳过 ts={ts}: 角点检测失败 "
                      f"(svc={'OK' if ca is not None else 'X'} "
                      f"d455={'OK' if cb is not None else 'X'})")
                continue
            ok_a, rva, tva = cv2.fisheye.solvePnP(
                objp.reshape(-1, 1, 3).astype(np.float64),
                ca.reshape(-1, 1, 2).astype(np.float64), self.K_svc, self.dist_svc)
            ok_b, rvb, tvb = cv2.solvePnP(objp, cb, self.K_d455, self.dist_d455)
            if not (ok_a and ok_b):
                print(f"  跳过 ts={ts}: PnP 失败")
                continue
            corners_a_list.append(ca.astype(np.float64))
            corners_b_list.append(cb.astype(np.float64))
            rvecs_a.append(rva.ravel()); tvecs_a.append(tva.ravel())
            rvecs_b.append(rvb.ravel()); tvecs_b.append(tvb.ravel())
            used_pairs.append((pa, pb, ts))

        N = len(used_pairs)
        print(f"  有效配对: {N}")
        if N == 0:
            raise RuntimeError("无有效配对，请检查棋盘参数或图像")

        # ---- 诊断 + 第一阶段离群点剔除 ----
        print("\n  [诊断] 单相机 PnP 重投影误差 (BA 前):")
        pnp_a_rms, pnp_b_rms = [], []
        for i in range(N):
            proj_a, _ = cv2.fisheye.projectPoints(
                objp.reshape(-1, 1, 3).astype(np.float64),
                rvecs_a[i], tvecs_a[i], self.K_svc, self.dist_svc)
            ea = np.sqrt(np.sum((proj_a.reshape(-1, 2) - corners_a_list[i])**2, axis=1))
            proj_b, _ = cv2.projectPoints(objp, rvecs_b[i], tvecs_b[i], self.K_d455, self.dist_d455)
            eb = np.sqrt(np.sum((proj_b.reshape(-1, 2) - corners_b_list[i])**2, axis=1))
            pnp_a_rms.append(float(np.sqrt(np.mean(ea**2))))
            pnp_b_rms.append(float(np.sqrt(np.mean(eb**2))))
            print(f"    对 {i+1}: SVC mean={ea.mean():.3f} max={ea.max():.3f} | "
                  f"D455 mean={eb.mean():.3f} max={eb.max():.3f}")

        keep_idx = [i for i in range(N)
                    if pnp_a_rms[i] < self.SVC_KEEP_RMS and pnp_b_rms[i] < self.D455_KEEP_RMS]
        drop_idx = [i for i in range(N) if i not in keep_idx]
        if drop_idx:
            print(f"\n  [剔除] 离群点对 (SVC>{self.SVC_KEEP_RMS}px 或 D455>{self.D455_KEEP_RMS}px): "
                  f"{[i+1 for i in drop_idx]}")
            corners_a_list = [corners_a_list[i] for i in keep_idx]
            corners_b_list = [corners_b_list[i] for i in keep_idx]
            rvecs_a = [rvecs_a[i] for i in keep_idx]
            tvecs_a = [tvecs_a[i] for i in keep_idx]
            rvecs_b = [rvecs_b[i] for i in keep_idx]
            tvecs_b = [tvecs_b[i] for i in keep_idx]
            used_pairs = [used_pairs[i] for i in keep_idx]
            N = len(keep_idx)
            print(f"  剩余有效配对: {N}")

        # ---- 4. 相对外参初值 (SVC -> D455) ----
        print("\n" + "=" * 60)
        print("4. 相对外参初值 (逐对 PnP, SVC -> D455)")
        print("=" * 60)
        R_rels, t_rels = [], []
        for i in range(N):
            R_a, _ = cv2.Rodrigues(rvecs_a[i])
            R_b, _ = cv2.Rodrigues(rvecs_b[i])
            R_rel = R_b @ R_a.T
            t_rel = tvecs_b[i] - R_rel @ tvecs_a[i]
            R_rels.append(R_rel); t_rels.append(t_rel)
            print(f"  对 {i+1}: t=[{t_rel[0]:8.4f} {t_rel[1]:8.4f} {t_rel[2]:8.4f}] m")

        quats = [Rotation.from_matrix(R).as_quat() for R in R_rels]
        q_mean = np.mean(quats, axis=0); q_mean /= np.linalg.norm(q_mean)
        R_init = Rotation.from_quat(q_mean).as_matrix()
        t_init = np.median(np.array(t_rels), axis=0)
        print(f"\n  初值 t = [{t_init[0]:.4f} {t_init[1]:.4f} {t_init[2]:.4f}] m")

        # ---- 5. 联合 BA (Cauchy 鲁棒损失 + 两阶段离群点剔除) ----
        print("\n" + "=" * 60)
        print("5. 联合 BA (Cauchy 鲁棒损失 + 两阶段离群点剔除)")
        print("=" * 60)

        # Pass 1: lm BA 全集
        res, R_opt, t_opt, rms_opt, per_pair_rms = self._run_ba(
            N, corners_a_list, corners_b_list, rvecs_a, tvecs_a,
            R_init, t_init,
            self.K_svc, self.dist_svc, self.K_d455, self.dist_d455,
            objp, use_cauchy=False)
        print(f"  [Pass 1] lm BA: 总 RMS = {rms_opt:.4f} px (N={N})")
        print(f"           t = [{t_opt[0]:.4f} {t_opt[1]:.4f} {t_opt[2]:.4f}] m")
        print(f"           每对 RMS: {[f'{x:.3f}' for x in per_pair_rms]}")

        # Pass 2: 剔除残差 > 1.5 * 中位数 的对, 用 Cauchy 精化
        median_rms = float(np.median(per_pair_rms))
        reject_threshold = self.REJECT_RATIO * median_rms
        keep_idx2 = [i for i, r in enumerate(per_pair_rms) if r <= reject_threshold]
        drop_idx2 = [i for i, r in enumerate(per_pair_rms) if r > reject_threshold]
        if drop_idx2 and len(keep_idx2) >= self.MIN_KEEP:
            print(f"\n  [Pass 2] 剔除离群对 (RMS > {reject_threshold:.3f}px = "
                  f"{self.REJECT_RATIO}*中位数 {median_rms:.3f}): "
                  f"对 {[i+1 for i in drop_idx2]}")
            c_a2 = [corners_a_list[i] for i in keep_idx2]
            c_b2 = [corners_b_list[i] for i in keep_idx2]
            rv_a2 = [rvecs_a[i] for i in keep_idx2]
            tv_a2 = [tvecs_a[i] for i in keep_idx2]
            res, R_opt, t_opt, rms_opt, per_pair_rms = self._run_ba(
                len(keep_idx2), c_a2, c_b2, rv_a2, tv_a2,
                R_opt, t_opt,
                self.K_svc, self.dist_svc, self.K_d455, self.dist_d455,
                objp, use_cauchy=True)
            N = len(keep_idx2)
            corners_a_list = c_a2
            corners_b_list = c_b2
            rvecs_a = rv_a2
            tvecs_a = tv_a2
            used_pairs = [used_pairs[i] for i in keep_idx2]
            print(f"  [Pass 2] Cauchy BA: 总 RMS = {rms_opt:.4f} px (N={N})")
            print(f"           t = [{t_opt[0]:.4f} {t_opt[1]:.4f} {t_opt[2]:.4f}] m")
            print(f"           每对 RMS: {[f'{x:.3f}' for x in per_pair_rms]}")
        else:
            print(f"  [Pass 2] 无离群点需剔除 (阈值 {reject_threshold:.3f}px, 保留全部 {N} 对)")

        self.rvec_opt = res.x[:3]
        self.R_opt, _ = cv2.Rodrigues(self.rvec_opt)
        self.t_opt = t_opt
        self.rms_opt = rms_opt
        self.n_pairs = N

        # ---- 6. 分相机重投影误差 ----
        print("\n" + "=" * 60)
        print("6. 重投影误差验证")
        print("=" * 60)
        off_r = 6; off_t = 6 + 3 * N
        err_a_all, err_b_all = [], []
        self.pair_err_lines = []
        for i in range(N):
            rvec_i = res.x[off_r + 3*i : off_r + 3*i + 3]
            tvec_i = res.x[off_t + 3*i : off_t + 3*i + 3]
            proj_a, _ = cv2.fisheye.projectPoints(
                objp.reshape(-1, 1, 3).astype(np.float64),
                rvec_i, tvec_i, self.K_svc, self.dist_svc)
            ea = np.sqrt(np.sum((proj_a.reshape(-1, 2) - corners_a_list[i])**2, axis=1))
            R_i, _ = cv2.Rodrigues(rvec_i)
            R_b = R_opt @ R_i
            t_b = (R_opt @ tvec_i.reshape(3, 1) + t_opt.reshape(3, 1)).ravel()
            rvec_b, _ = cv2.Rodrigues(R_b)
            proj_b, _ = cv2.projectPoints(objp, rvec_b, t_b, self.K_d455, self.dist_d455)
            eb = np.sqrt(np.sum((proj_b.reshape(-1, 2) - corners_b_list[i])**2, axis=1))
            err_a_all.append(ea); err_b_all.append(eb)
            line = (f"  对 {i+1}: SVC mean={ea.mean():.3f} max={ea.max():.3f} | "
                    f"D455 mean={eb.mean():.3f} max={eb.max():.3f}")
            print(line); self.pair_err_lines.append(line)
        ea_all = np.concatenate(err_a_all); eb_all = np.concatenate(err_b_all)
        self.svc_mean  = float(ea_all.mean());  self.svc_rms  = float(np.sqrt(np.mean(ea_all**2)))
        self.d455_mean = float(eb_all.mean()); self.d455_rms = float(np.sqrt(np.mean(eb_all**2)))
        print(f"  总体 SVC : mean={self.svc_mean:.3f} rms={self.svc_rms:.3f}")
        print(f"  总体 D455: mean={self.d455_mean:.3f} rms={self.d455_rms:.3f}")

        # ---- 7. 输出结果 ----
        print("\n" + "=" * 60)
        print("7. 外参标定结果 (SVC -> D455) - 第 2 组")
        print("=" * 60)
        print(f"\nR =\n{self.R_opt}")
        print(f"\nt = [{self.t_opt[0]:.6f} {self.t_opt[1]:.6f} {self.t_opt[2]:.6f}] m")
        euler = Rotation.from_matrix(self.R_opt).as_euler('xyz', degrees=True)
        print(f"\n欧拉角 (deg, xyz/RPY) = {euler}")

        np.savez(
            self.out_npz,
            R=self.R_opt, t=self.t_opt, rvec=self.rvec_opt,
            K_d455=self.K_d455, dist_d455=self.dist_d455,
            K_svc=self.K_svc, dist_svc=self.dist_svc,
            rms=self.rms_opt, n_pairs=self.n_pairs)
        print(f"\n✅ 已保存: {self.out_npz}")

        with open(self.out_txt, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("SVC(针孔+Kb 鱼眼) -> D455(针孔+Plumb-Bob) 外参标定结果 - 第 2 组\n")
            f.write("P_d455 = R @ P_svc + t\n")
            f.write("=" * 60 + "\n\n")

            f.write("[相机内参]\n")
            f.write(f"SVC  K (fx,fy,cx,cy) = "
                    f"{self.K_svc[0,0]:.4f}, {self.K_svc[1,1]:.4f}, "
                    f"{self.K_svc[0,2]:.4f}, {self.K_svc[1,2]:.4f}\n")
            f.write(f"SVC  dist (Kb) = {self.dist_svc.ravel()}\n")
            f.write(f"SVC  Kb 内参标定 RMS = {self.rms_svc_intrinsic:.4f} px\n")
            f.write(f"D455 K (fx,fy,cx,cy) = "
                    f"{self.K_d455[0,0]:.4f}, {self.K_d455[1,1]:.4f}, "
                    f"{self.K_d455[0,2]:.4f}, {self.K_d455[1,2]:.4f}\n")
            f.write(f"D455 dist (Plumb-Bob) = {self.dist_d455.ravel()}\n\n")

            f.write("[外参]\n")
            f.write(f"R =\n{self.R_opt}\n")
            f.write(f"t = [{self.t_opt[0]:.6f} {self.t_opt[1]:.6f} {self.t_opt[2]:.6f}] m\n")
            f.write(f"欧拉角 (deg, xyz/RPY) = {euler}\n")
            f.write(f"rvec (Rodrigues) = {self.rvec_opt.ravel()}\n\n")

            f.write("[标定规模]\n")
            f.write(f"有效配对数 N = {self.n_pairs}\n")
            f.write(f"棋盘格内角点 = {self.checkerboard}, 每格边长 = {self.square_size} m\n\n")

            f.write("[重投影误差]\n")
            f.write(f"BA 优化后 总 RMS = {self.rms_opt:.4f} px\n")
            for line in self.pair_err_lines:
                f.write(line + "\n")
            f.write(f"总体 SVC : mean={self.svc_mean:.3f} rms={self.svc_rms:.3f} px\n")
            f.write(f"总体 D455: mean={self.d455_mean:.3f} rms={self.d455_rms:.3f} px\n")
        print(f"✅ 已保存: {self.out_txt}")


# ==================== 主入口 ====================
if __name__ == "__main__":
    calib = ExtrinsicCalibrator(
        d455_intrinsics_path="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/data/标定图像2/intrinsics.txt",
        svc_intrinsics_path="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/data/car_intrinsics.txt",
        d455_img_dir="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/data/标定图像2/d455",
        svc_img_dir="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/data/标定图像2/car/crop",
        checkerboard=(9, 6),
        square_size=0.037,
        out_npz="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/code/photoPre/extrinsic_svc_d455_set2.npz",
        out_txt="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/code/photoPre/extrinsic_svc_d455_set2.txt",
    )
    calib.run()
