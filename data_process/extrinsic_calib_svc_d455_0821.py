#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVC(针孔+Kb 鱼眼) -> D455(针孔+Plumb-Bob) 外参标定 - 0821 数据
数据: /media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/svcdata/0821/deal/calib
复用 extrinsic_calib_svc_d455_set2.ExtrinsicCalibrator, 仅配对方式不同:
  svc_pad: scene_N_CALIB_*.png   d455: scene_N_*.png   -> 按 scene N 配对
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extrinsic_calib_svc_d455_set2 import ExtrinsicCalibrator

BASE = "/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/svcdata/0821/deal/calib"


class ExtrinsicCalibrator0821(ExtrinsicCalibrator):
    def pair_images(self, d455_dir, svc_dir):
        def scene_no(name):
            m = re.match(r"scene_(\d+)_", Path(name).stem)
            return int(m.group(1)) if m else None
        d455 = {scene_no(f): os.path.join(d455_dir, f)
                for f in os.listdir(d455_dir) if scene_no(f) is not None}
        svc = {scene_no(f): os.path.join(svc_dir, f)
               for f in os.listdir(svc_dir) if scene_no(f) is not None}
        return [(d455[n], svc[n], n) for n in sorted(d455) if n in svc]


if __name__ == "__main__":
    calib = ExtrinsicCalibrator0821(
        d455_intrinsics_path=f"{BASE}/intrinsics.txt",
        svc_intrinsics_path=f"{BASE}/ex_svc_right.json",
        d455_img_dir=f"{BASE}/d455",
        svc_img_dir=f"{BASE}/svc_pad",
        checkerboard=(9, 6),
        square_size=0.037,
        out_npz="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/code/photoPre/extrinsic_svc_d455_0821.npz",
        out_txt="/media/shuyuan/a477aba2-be55-498b-b0b7-dafef32f2c24/code/photoPre/extrinsic_svc_d455_0821.txt",
    )
    calib.run()
