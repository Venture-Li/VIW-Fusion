#!/usr/bin/env python3
"""Evaluate world landmarks against depth reprojected into the processed SVC camera.
Calibration JSON is explicit: P_processed = R_depth_to_processed @ P_depth + t.
Timestamp offsets are added to file timestamps to express them on the pose/image clock.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def matrix(path, key):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f'Cannot open {path}')
    value = fs.getNode(key).mat()
    fs.release()
    if value is None:
        raise ValueError(f'{key} missing from {path}')
    return value


def image_index(directory):
    index = directory / 'data.csv'
    entries = []
    if index.exists():
        with index.open() as f:
            for row in csv.reader(f):
                if not row or row[0].lstrip().startswith('#'):
                    continue
                try:
                    stamp = int(row[0]) * 1e-9
                except ValueError:
                    continue
                name = row[1].strip()
                path = directory / 'data' / name
                if not path.exists():
                    path = directory / name
                entries.append((stamp, path))
    else:
        for path in directory.iterdir():
            if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            # Converter preserves a final millisecond timestamp; bag-export names are ns.
            token = path.stem.split('_')[-1]
            try:
                value = int(token)
            except ValueError:
                continue
            entries.append((value * (1e-9 if value >= 10**17 else 1e-3), path))
    if not entries:
        raise ValueError(f'No timestamped images in {directory}')
    return entries


def nearest(entries, timestamp, offset, tolerance):
    stamp, path = min(entries, key=lambda item: abs(item[0] + offset - timestamp))
    delta = stamp + offset - timestamp
    if abs(delta) > tolerance:
        raise ValueError(f'{path}: corrected timestamp mismatch {delta * 1000:.3f} ms')
    return path, delta


def rotation(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid pose quaternion')
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def register_depth(depth, kd, kp, r, t, shape):
    """Forward projection with nearest-surface z buffering; leave holes invalid."""
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    rays = np.linalg.inv(kd) @ np.vstack((x, y, np.ones(len(x))))
    p = r @ (rays * depth[y, x]) + t[:, None]
    keep = np.isfinite(p).all(axis=0) & (p[2] > 0)
    p = p[:, keep]
    uv = kp @ p
    uv = np.rint(uv[:2] / uv[2]).astype(np.int64)
    h, w = shape
    keep = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
    result = np.full(h*w, np.inf)
    np.minimum.at(result, uv[1, keep]*w + uv[0, keep], p[2, keep])
    result[~np.isfinite(result)] = 0
    return result.reshape(h, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Exact YAML used by this SLAM run')
    parser.add_argument('--calibration', type=Path, help='Optional JSON override; repository D455/SVC calibration is used by default')
    parser.add_argument('--camera-extrinsics', type=Path, help='Final body_T_cam0 YAML if optimized online')
    parser.add_argument('--frame-index', type=int, default=2, help='Zero-based pose row, default third row')
    parser.add_argument('--max-depth', type=float, default=3.)
    parser.add_argument('--max-time-diff', type=float, default=.05, help='Seconds; no motion compensation')
    args = parser.parse_args()
    # D455.txt: Color -> Depth. The processed SVC image has SVC origin and
    # Color-aligned axes, so depth -> processed is the inverse Color->Depth
    # transform. Values below are derived from repository calibration files.
    r_cd = np.array([[0.999994, -0.0000966545, -0.00353483],
                     [0.0000784609, 0.999987, -0.00514674],
                     [0.00353528, 0.00514643, 0.999981]])
    t_cd = np.array([0.0590972937643528, -0.0000248234175, -0.000321907602483407])
    t_sc = np.array([-0.021910, 0.123421, 0.096283])
    cal = {
        'K_depth': [[386.724945, 0, 323.348145], [0, 386.724945, 233.675964], [0, 0, 1]],
        'R_depth_to_processed': r_cd.T.tolist(),
        't_depth_to_processed_m': (-r_cd.T @ (r_cd @ t_sc + t_cd)).tolist(),
        'depth_scale_m': 0.001,
        'depth_time_offset_s': 0.0,
        'svc_time_offset_s': 0.0,
        'source': 'data_process/D455.txt + extrinsic_svc_d455_1018_fused.txt'
    }
    if args.calibration:
        cal.update(json.loads(args.calibration.read_text()))
    required = ['K_depth', 'R_depth_to_processed', 't_depth_to_processed_m',
                'depth_scale_m', 'depth_time_offset_s', 'svc_time_offset_s']
    for key in required:
        if key not in cal or cal[key] is None:
            raise ValueError(f'Calibration requires confirmed {key}')
    kd = np.asarray(cal['K_depth'], dtype=float)
    r = np.asarray(cal['R_depth_to_processed'], dtype=float)
    t = np.asarray(cal['t_depth_to_processed_m'], dtype=float)
    if kd.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
        raise ValueError('Calibration matrix/vector shape invalid')
    if not np.isfinite(kd).all() or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError('Non-finite calibration')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1., atol=1e-4):
        raise ValueError('Depth-to-processed rotation must be a proper rotation')
    if float(cal['depth_scale_m']) <= 0 or args.max_time_diff <= 0:
        raise ValueError('Depth scale and time tolerance must be positive')
    fs = cv2.FileStorage(str(args.config), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError('Cannot open SLAM configuration')
    optimized = int(fs.getNode('estimate_extrinsic').real()) != 0
    intrinsics = args.config.parent / fs.getNode('cam0_calib').string()
    fs.release()
    if optimized and args.camera_extrinsics is None:
        raise ValueError('Camera optimized online: provide final --camera-extrinsics, not initial config')
    tic = matrix(args.camera_extrinsics or args.config, 'body_T_cam0')
    fs = cv2.FileStorage(str(intrinsics), cv2.FILE_STORAGE_READ)
    if fs.getNode('model_type').string() != 'PINHOLE':
        raise ValueError('Processed camera must be PINHOLE')
    distortion = fs.getNode('distortion_parameters')
    if any(abs(distortion.getNode(k).real()) > 1e-12 for k in ('k1', 'k2', 'p1', 'p2')):
        raise ValueError('Expected rectified processed camera with zero distortion')
    proj = fs.getNode('projection_parameters')
    fx, fy, cx, cy = [proj.getNode(k).real() for k in ('fx', 'fy', 'cx', 'cy')]
    kp = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    fs.release()
    poses = np.loadtxt(args.scene_dir / 'log/pose_output.txt', ndmin=2)
    row = poses[args.frame_index]
    stamp, pos, rw = row[0], row[1:4], rotation(row[4:8])
    image_path, cam_dt = nearest(image_index(args.data_root / 'svc_processed'), stamp,
                                  cal['svc_time_offset_s'], args.max_time_diff)
    depth_path, depth_dt = nearest(image_index(args.data_root / 'depth0'), stamp,
                                    cal['depth_time_offset_s'], args.max_time_diff)
    image = cv2.imread(str(image_path))
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if image is None or raw is None or raw.ndim != 2:
        raise ValueError('Cannot read color/depth image or depth is not single-channel')
    aligned = register_depth(raw.astype(float)*cal['depth_scale_m'], kd, kp, r, t, image.shape[:2])
    np.save(args.scene_dir / 'registered_depth_svc.npy', aligned)
    colored = cv2.applyColorMap(np.uint8(np.clip(aligned/args.max_depth, 0, 1)*255), cv2.COLORMAP_JET)
    overlay = image.copy()
    valid = aligned > 0
    overlay[valid] = cv2.addWeighted(image, .6, colored, .4, 0)[valid]
    cv2.imwrite(str(args.scene_dir / 'color_depth_overlay.png'), overlay)
    comparisons = []
    h, w = aligned.shape
    with (args.scene_dir / 'log/points_output.txt').open() as f:
        for line in f:
            values = np.fromstring(line, sep=' ')
            if len(values) < 3:
                continue
            if len(values) > 3 and not np.any(np.abs(values[3:]-stamp) < 1e-4):
                continue
            p = tic[:3, :3].T @ (rw.T @ (values[:3]-pos)-tic[:3, 3])
            if not np.isfinite(p).all() or not 0 < p[2] < args.max_depth:
                continue
            uv = kp @ p
            u, v = np.rint(uv[:2]/uv[2]).astype(int)
            if not (1 <= u < w-1 and 1 <= v < h-1):
                continue
            gt = aligned[v, u]
            neighborhood = aligned[v-1:v+2, u-1:u+2]
            surface = neighborhood[neighborhood > 0]
            # Reject holes and discontinuities; never fill holes with estimated SLAM depth.
            if not 0 < gt < args.max_depth or len(surface) < 5:
                continue
            if np.ptp(surface) > max(.05, .03*gt):
                continue
            comparisons.append((u, v, p[2], gt, abs(p[2]-gt), abs(p[2]-gt)/gt))
    with (args.scene_dir / 'depth_comparison.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['u_s', 'v_s', 'est_depth_svc(m)', 'gt_depth_svc(m)', 'abs_error(m)', 'rel_error'])
        writer.writerows(comparisons)
    if not comparisons:
        raise ValueError('No valid depth correspondences; inspect registered overlay and timestamp diagnostics')
    a = np.asarray(comparisons)
    slope = np.linalg.lstsq(a[:, 3:4], a[:, 2], rcond=None)[0][0]
    result = dict(timestamp=float(stamp), valid_points=len(a), slope=float(slope),
                  rel_err_mean=float(a[:, 5].mean()), rel_err_median=float(np.median(a[:, 5])),
                  rel_err_std=float(a[:, 5].std()), svc_time_difference_s=cam_dt,
                  depth_time_difference_s=depth_dt, image=str(image_path), depth=str(depth_path),
                  config=str(args.config.resolve()), calibration=cal,
                  note='Static extrinsics only; no inter-frame motion compensation. Occlusion checks are approximate.')
    (args.scene_dir / 'depth_eval_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
