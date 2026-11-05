#!/usr/bin/env python3
"""Visualize D455 depth reprojected into the processed SVC image."""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

K = np.array([[386.724945, 0, 323.348145],
              [0, 386.724945, 233.675964],
              [0, 0, 1]], dtype=np.float64)
# D455 Color -> Depth, from data_process/D455.txt
R_cd = np.array([[0.999994, -0.0000966545, -0.00353483],
                 [0.0000784609, 0.999987, -0.00514674],
                 [0.00353528, 0.00514643, 0.999981]], dtype=np.float64)
t_cd = np.array([0.0590972937643528, -0.0000248234175,
                 -0.000321907602483407], dtype=np.float64)
# SVC -> D455 Color, from extrinsic_svc_d455_1018_fused.txt
R_sc = np.array([[0.99176726, 0.03108473, -0.12422338],
                 [-0.03676192, 0.99836928, -0.04367313],
                 [0.12266324, 0.04788027, 0.99129270]], dtype=np.float64)
t_sc = np.array([-0.021910, 0.123421, 0.096283], dtype=np.float64)
# Processed SVC has SVC origin and D455 Color-aligned axes.
R_dp = R_cd.T
# P_depth = R_cd (P_processed + t_sc) + t_cd.
t_dp = -R_cd.T @ (R_cd @ t_sc + t_cd)


def images(folder):
    folder = Path(folder)
    csv_path = folder / 'data.csv'
    out = []
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.reader(f):
                if not row or row[0].startswith('#'):
                    continue
                try: stamp = int(row[0]) * 1e-9
                except ValueError: continue
                p = folder / 'data' / row[1]
                if p.exists(): out.append((stamp, p))
    else:
        for p in folder.glob('*'):
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'): continue
            try: stamp = int(p.stem.split('_')[-1]) * (1e-9 if len(p.stem.split('_')[-1]) >= 17 else 1e-3)
            except ValueError: continue
            out.append((stamp, p))
    if not out: raise RuntimeError(f'No timestamped images found in {folder}')
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', type=Path, required=True, help='scene_N data directory')
    ap.add_argument('--timestamp', type=float, help='seconds; default: middle SVC frame')
    ap.add_argument('--max-depth', type=float, default=8.0)
    ap.add_argument('--alpha', type=float, default=.55)
    ap.add_argument('--output', type=Path, help='output PNG; default scene/alignment_overlay.png')
    args = ap.parse_args()
    svc = images(args.scene / 'svc_processed')
    dep = images(args.scene / 'depth0')
    target = args.timestamp if args.timestamp is not None else svc[len(svc)//2][0]
    svc_ts, svc_path = min(svc, key=lambda x: abs(x[0]-target))
    dep_ts, dep_path = min(dep, key=lambda x: abs(x[0]-target))
    color = cv2.imread(str(svc_path), cv2.IMREAD_COLOR)
    raw = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
    if color is None or raw is None or raw.ndim != 2: raise RuntimeError('Cannot read SVC/depth image')
    depth = raw.astype(np.float64) * .001
    h, w = color.shape[:2]
    yy, xx = np.indices(depth.shape)
    valid = (depth > 0) & (depth < args.max_depth)
    rays = np.linalg.inv(K) @ np.vstack((xx[valid], yy[valid], np.ones(valid.sum())))
    p_d = rays * depth[valid]
    p_p = R_dp @ p_d + t_dp[:, None]
    valid2 = p_p[2] > 0
    uv = K @ p_p[:, valid2]
    u = np.rint(uv[0]/uv[2]).astype(int)
    v = np.rint(uv[1]/uv[2]).astype(int)
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[keep]
    v = v[keep]
    z = p_p[2, valid2][keep]
    # z-buffer: nearest depth wins at each processed-SVC pixel.
    registered = np.zeros((h, w), np.float32)
    order = np.argsort(z)
    for j in order:
        if registered[v[j], u[j]] == 0: registered[v[j], u[j]] = z[j]
    depth_color = cv2.applyColorMap(np.uint8(np.clip(registered/args.max_depth, 0, 1)*255), cv2.COLORMAP_JET)
    mask = registered > 0
    overlay = color.copy()
    overlay[mask] = cv2.addWeighted(color, 1-args.alpha, depth_color, args.alpha, 0)[mask]
    # Draw sparse projected points in green for an easier geometric check.
    # The projected depth lattice is not scene geometry. Draw only sparse
    # samples so it does not mask the image and look like false contours.
    step = max(1, len(u)//800)
    for x, y in zip(u[::step], v[::step]): cv2.circle(overlay, (int(x), int(y)), 1, (0,255,0), -1)
    cv2.putText(overlay, f'SVC {svc_ts:.6f}  depth {dep_ts:.6f}  dt={(dep_ts-svc_ts)*1000:.1f} ms',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
    output = args.output or (args.scene / 'alignment_overlay.png')
    cv2.imwrite(str(output), overlay)
    cv2.imwrite(str(args.scene / 'registered_depth_svc.png'), np.uint16(np.clip(registered*1000, 0, 65535)))
    depth_view = np.uint8(np.clip(registered/args.max_depth, 0, 1)*255)
    depth_view[~mask] = 0
    cv2.imwrite(str(args.scene / 'registered_depth_svc_view.png'),
                cv2.applyColorMap(depth_view, cv2.COLORMAP_JET))
    print(f'SVC: {svc_path} ({svc_ts:.6f})')
    print(f'Depth: {dep_path} ({dep_ts:.6f}), dt={(dep_ts-svc_ts)*1000:.2f} ms')
    print(f'Projected depth pixels: {int(mask.sum())}/{h*w} ({100*mask.mean():.1f}%)')
    print(f'Overlay: {output}')
    print(f'Depth visualization: {args.scene / "registered_depth_svc_view.png"}')

if __name__ == '__main__': main()
