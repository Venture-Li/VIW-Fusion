#!/usr/bin/env python3
"""
Extract /vins_estimator/image_track frames from track.bag files to PNG images.
Images are saved in the same directory as the source bag file.

Usage:
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate da
    python /home/nio/vscode/python_project_backup/VIW-Fusion/run/extract_track_images.py

Or process a single bag:
    python /home/nio/vscode/python_project_backup/VIW-Fusion/run/extract_track_images.py /path/to/track.bag
"""

import os
import sys
import struct
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader


def parse_image_msg(rawdata):
    """
    Parse a ROS1 serialized sensor_msgs/Image message.
    Returns (encoding, width, height, image_numpy_array) or None on failure.
    """
    data = rawdata
    pos = 0

    try:
        # Header: uint32 seq
        seq = struct.unpack_from('<I', data, pos)[0]; pos += 4
        # time stamp: uint32 sec + uint32 nsec
        stamp_sec = struct.unpack_from('<I', data, pos)[0]; pos += 4
        stamp_nsec = struct.unpack_from('<I', data, pos)[0]; pos += 4
        # string frame_id
        frame_id_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
        frame_id = data[pos:pos + frame_id_len].decode('ascii'); pos += frame_id_len

        # uint32 height, width
        height = struct.unpack_from('<I', data, pos)[0]; pos += 4
        width = struct.unpack_from('<I', data, pos)[0]; pos += 4

        # string encoding
        enc_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
        encoding = data[pos:pos + enc_len].decode('ascii'); pos += enc_len

        # uint8 is_bigendian
        is_bigendian = struct.unpack_from('<B', data, pos)[0]; pos += 1

        # uint32 step
        step = struct.unpack_from('<I', data, pos)[0]; pos += 4

        # uint8[] data: uint32 length + raw bytes
        data_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
        img_data = data[pos:pos + data_len]

        # Decode image based on encoding
        if encoding == 'bgr8':
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
        elif encoding == 'rgb8':
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding == 'mono8' or encoding == '8UC1':
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width)
        elif encoding == '8UC3':
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
        else:
            print(f"  WARNING: unknown encoding '{encoding}', trying as bgr8")
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)

        return encoding, width, height, img

    except Exception as e:
        print(f"  ERROR parsing message: {e}")
        return None


def extract_bag(bag_path):
    """Extract all image frames from a single track.bag."""
    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"SKIP: {bag_path} not found")
        return

    out_dir = bag_path.parent / f"{bag_path.stem}_images"
    out_dir.mkdir(exist_ok=True)

    print(f"Processing: {bag_path}")
    frame_idx = 0

    with Reader(str(bag_path)) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic != '/vins_estimator/image_track':
                continue

            result = parse_image_msg(rawdata)
            if result is None:
                continue

            encoding, width, height, img = result
            stamp_sec = timestamp // 10**9
            stamp_nsec = timestamp % 10**9

            out_name = f"{stamp_sec}.{stamp_nsec:09d}.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), img)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  {frame_idx} frames extracted...")

    print(f"  Done: {frame_idx} frames → {out_dir}")


def main():
    if len(sys.argv) > 1:
        # Single bag mode
        extract_bag(sys.argv[1])
    else:
        # Batch mode: find all track.bag under data directory
        data_base = Path("/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/data")
        if not data_base.exists():
            print(f"ERROR: data directory not found: {data_base}")
            sys.exit(1)

        bag_files = sorted(data_base.glob("scene_*/track.bag"))
        if not bag_files:
            print(f"No track.bag files found under {data_base}")
            sys.exit(1)

        print(f"Found {len(bag_files)} track.bag files")
        for bag in bag_files:
            extract_bag(bag)
            print()

    print("All done.")


if __name__ == "__main__":
    main()
